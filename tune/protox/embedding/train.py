import sys
import os
import yaml
import json
import random
import argparse
import shutil
import tqdm
import numpy as np
import logging
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import quantile_transform
import pandas as pd
from pathlib import Path
import time
import logging
from datetime import datetime
from functools import partialmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils import logging_presets
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance

import ray
from ray.tune import with_resources, with_parameters, Tuner, TuneConfig
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune import Trainable, SyncConfig
from ray.air import session, RunConfig, FailureConfig
from hyperopt import hp
import hyperopt.pyll

from embeddings.loss import COST_COLUMNS, CostLoss, get_bias_fn
from embeddings.vae import gen_vae_collate, VAE, VAELoss
from embeddings.trainer import VAETrainer, StratifiedRandomSampler
from embeddings.utils import f_unpack_dict, parse_hyperopt_config

from envs.workload import Workload
from envs.spaces import IndexSpace
from envs.spaces.index_policy import IndexRepr


def _fetch_index_parameters(data):
    tables = data["mythril"]["tables"]
    attributes = data["mythril"]["attributes"]
    query_spec = data["mythril"]["query_spec"]
    workload = Workload(tables, attributes, query_spec, pid=None)
    att_usage = workload.process_column_usage()

    space = IndexSpace(
        "wolp",
        tables,
        max_num_columns=0,
        index_repr=IndexRepr.ONE_HOT.name,
        seed=0,
        latent_dim=0,
        attributes_overwrite=att_usage)
    space._build_mapping(att_usage)
    max_cat_features = max(len(tables), space.max_num_columns + 1) # +1 for the one hot encoding.
    max_attrs = space.max_num_columns + 1 # +1 to account for the table index.
    return max_attrs, max_cat_features, att_usage, space.class_mapping


def _load_input_data(input_file, train_size, max_attrs, require_cost, seed):
    # Load the input data.
    columns = []
    columns += ["tbl_index", "idx_class"]
    columns += [f"col{c}" for c in range(max_attrs - 1)]
    if require_cost:
        columns += COST_COLUMNS

    df = pd.read_parquet(input_file, columns=columns)
    num_classes = df.idx_class.max() + 1

    # Get the y's and the x's.
    targets = (COST_COLUMNS + ["idx_class"]) if require_cost else ["idx_class"]
    y = df[targets].values
    df.drop(columns=COST_COLUMNS + ["idx_class"], inplace=True, errors="ignore")
    x = df.values
    del df
    gc.collect()
    gc.collect()

    if train_size == 1:
        train_dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y))
        del x
        gc.collect()
        gc.collect()
        return train_dataset, y, y[:, -1], None, num_classes

    # Perform the train test split.
    train_x, val_x, train_y, val_y = train_test_split(
        x, y,
        test_size=1 - train_size,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=y[:, -1])
    del x
    del y
    gc.collect()
    gc.collect()

    # Form the tensor datasets.
    train_dataset = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
    val_dataset = TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y))
    del val_x
    del val_y
    del train_x
    gc.collect()
    gc.collect()
    logging.info("Train Dataset Size: %s", len(train_dataset))
    return train_dataset, train_y, train_y[:, -1], val_dataset, num_classes


def _create_vae_model(config, max_attrs, max_cat_features):
    cat_input = max_attrs * max_cat_features

    assert config["act"] in ["relu", "mish"]
    assert config["mean_output_act"] in ["tanh_squash", "sigmoid"]

    mean_output_act = {
        "sigmoid": nn.Sigmoid,
    }[config["mean_output_act"]]

    torch.set_float32_matmul_precision("high")
    model = VAE(
        max_categorical=max_cat_features,
        input_dim=cat_input,
        hidden_sizes=list(config["hidden_sizes"]),
        latent_dim=config["latent_dim"],
        act=nn.ReLU if config["act"] == "relu" else nn.Mish,
        bias_init=config["bias_init"],
        weight_init=config["weight_init"],
        weight_uniform=config["weight_uniform"],
        mean_output_act=mean_output_act,
        output_scale=config.get("output_scale", 1.0),
    )

    return model


def construct_epoch_end(val_dl, config, hooks, model_folder):
    def epoch_end(trainer, *args, **kwargs):
        save_interval = config.get("save_every", 1)
        if (trainer.epoch - 1) % save_interval == 0:
            # Save.
            mf = Path(model_folder) / f"epoch{trainer.epoch}"
            mf.mkdir(parents=True, exist_ok=True)
            hooks.save_models(trainer, str(mf), str(trainer.epoch))

        force = kwargs.get("force", False)
        suppress = kwargs.get("suppress", False)

        if force:
            total_metric_loss = []
            total_recon_loss = []
            with torch.no_grad():
                # Switch to eval mode.
                trainer.switch_eval()

                pbar = None if suppress else tqdm.tqdm(total=len(val_dl))
                for i, curr_batch in enumerate(val_dl):
                    # Get the losses.
                    trainer.calculate_loss(curr_batch)
                    if isinstance(trainer.losses["metric_loss"], torch.Tensor):
                        total_metric_loss.append(trainer.losses["metric_loss"].item())
                    else:
                        total_metric_loss.append(trainer.losses["metric_loss"])
                    total_recon_loss.append(trainer.last_recon_loss)

                    if pbar is not None:
                        pbar.set_description("total_recon=%.5f total_metric=%.5f" % (total_recon_loss[-1], total_metric_loss[-1]))
                        pbar.update(1)

                # Switch to train mode.
                trainer.switch_train()

        if force:
            return {
                "avg_metric": np.mean(total_metric_loss),
                "avg_recon": np.mean(total_recon_loss),
                "total_avg_loss": np.mean(total_metric_loss) + np.mean(total_recon_loss),
            }

    return epoch_end


def build_trainer(config, input_file, trial_dir, benchmark_config, train_size, dataloader_num_workers=0, disable_tqdm=False):
    max_cat_features = 0
    max_attrs = 0

    # Load the benchmark configuration.
    with open(benchmark_config, "r") as f:
        data = yaml.safe_load(f)
        max_attrs, max_cat_features, att_usage, class_mapping = _fetch_index_parameters(data)

    config["class_mapping"] = {}
    for (tbl, col), key in class_mapping.items():
        config["class_mapping"][str(key)] = {
            "relname": tbl,
            "ord_column": col,
        }

    # Device.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Get the datasets.
    train_dataset, train_y, idx_class, val_dataset, num_classes = _load_input_data(
        input_file,
        train_size,
        max_attrs,
        config["metric_loss_md"].get("require_cost", False),
        config["seed"])

    # Acquire the collation function.
    collate_fn = gen_vae_collate(max_cat_features)

    # Construct the models and optimizers.
    model = _create_vae_model(config, max_attrs, max_cat_features)
    model.to(device=device)

    # Trunk is the identity.
    trunk = nn.Sequential(nn.Identity())
    trunk.to(device=device)

    models = {"trunk": trunk, "embedder": model}
    optimizers = { "embedder_optimizer": torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]), }

    metric_loss = CostLoss(config["metric_loss_md"])
    # Default miner.
    tminers = {}

    # Define the loss functions.
    loss_funcs = {
        "metric_loss": metric_loss,
        "vae_loss": VAELoss(config["loss_fn"], max_attrs, max_cat_features),
    }

    loss_weights = {"metric_loss": config["metric_loss_weight"], "vae_loss": 1}

    # Define the sampler.
    sampler = StratifiedRandomSampler(idx_class, max_class=num_classes, batch_size=config["batch_size"], allow_repeats=True)

    # Define the tester hook.
    record_keeper, _, _ = logging_presets.get_record_keeper(f"{trial_dir}/logs", f"{trial_dir}/tboard")
    hooks = logging_presets.get_hook_container(record_keeper)
    model_folder = f"{trial_dir}/models"

    # Validation step loop.
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=4096, collate_fn=collate_fn)
    epoch_end = construct_epoch_end(val_dl, config, hooks, model_folder)

    def clip_grad():
        if config["grad_clip_amount"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_amount"])

    bias_fn = None
    if config["use_bias"]:
        bias_fn = get_bias_fn(config)

    # Build the trainer.
    return VAETrainer(
        disable_tqdm=disable_tqdm,
        bias_fn=bias_fn,
        models=models,
        optimizers=optimizers,
        batch_size=config["batch_size"],
        loss_funcs=loss_funcs,
        mining_funcs=tminers,
        dataset=train_dataset,
        sampler=sampler,
        iterations_per_epoch=config["iterations_per_epoch"] if config["iterations_per_epoch"] is not None else int(len(train_dataset) / config["batch_size"]),
        data_device=device,
        dtype=None,
        loss_weights=loss_weights,
        collate_fn=collate_fn,
        lr_schedulers=None,
        gradient_clippers={"embedder_grad_clipper": clip_grad},
        dataloader_num_workers=dataloader_num_workers,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=epoch_end,
    ), epoch_end


def create_train_parser(subparser):
    parser = subparser.add_parser("train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-data", type=str)
    parser.add_argument("--table-shape", action="store_true")
    parser.add_argument("--config", type=Path, default="embeddings/config.json")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--train-size", type=float, default=0.8)

    # Arguments for all models.
    parser.add_argument("--iterations-per-epoch", type=int)

    parser.add_argument("--ray-num-gpu", type=int, default=0)
    parser.add_argument("--max-concurrent", default=1, type=int)
    parser.add_argument("--mythril-dir", type=str)
    parser.add_argument("--num-threads", type=int, default=None)

    parser.add_argument("--gen-only", action="store_true", default=False)
    parser.add_argument("--dual-class", action="store_true", default=False)
    parser.add_argument("--inflate-ratio", type=int, default=1)
    parser.add_argument("--pad-min", type=int, default=None)
    parser.add_argument("--rebias", type=float, default=0)
    parser.add_argument("--manual", action="store_true", default=False)
    parser.set_defaults(func=execute_train)


def execute_manual(args, space):
    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = args.output_dir / f"embeddings_{dtime}"
    output_root.mkdir(parents=True, exist_ok=True)

    trial_num = 1
    total_trials = args.num_trials
    while trial_num <= total_trials:
        trial_dir = (output_root / f"trial{trial_num}")
        trial_dir.mkdir(parents=True, exist_ok=False)
        logging.info(f"Starting trial {trial_num}")

        config = hyperopt.pyll.stochastic.sample(space)
        config = f_unpack_dict(config)

        # Seed
        seed = np.random.randint(1, 1e8)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        config["seed"] = seed
        config["iterations_per_epoch"] = args.iterations_per_epoch

        if config.get("use_bias", False):
            if "bias_separation" in config and "addtl_bias_separation" in config and "output_scale" in config:
                # Do a hacky reconfigure.
                if config["output_scale"] > config["bias_separation"] + config["addtl_bias_separation"]:
                    config["output_scale"] = config["bias_separation"] + config["addtl_bias_separation"]
            config["metric_loss_md"]["output_scale"] = config["output_scale"]
        else:
            config["metric_loss_md"]["output_scale"] = config["output_scale"]

        logging.info(config)

        # Build trainer and train.
        trainer, epoch_end = build_trainer(
            config,
            f"{args.output_dir}/out.parquet",
            trial_dir,
            args.benchmark_config,
            args.train_size
        )

        # Dump the config that we are executing.
        with open(f"{trial_dir}/config", "w") as f:
            f.write(json.dumps(config, indent=4))

        trainer.train(num_epochs=config["num_epochs"])
        if trainer.failed:
            # Trainer has failed.
            with open(f"{trial_dir}/FAILED", "w") as f:
                if trainer.fail_msg is not None:
                    f.write(trainer.fail_msg)

            if trainer.fail_data is not None:
                torch.save(trainer.fail_data, f"{trial_dir}/fail_data.pth")
        else:
            loss = epoch_end(trainer, force=True)["total_avg_loss"]
            logging.info(f"Finished trial with {loss}.")
            trial_num += 1


def hpo_train(config, args):
    assert args is not None
    mythril_dir = os.path.expanduser(args["mythril_dir"])
    sys.path.append(mythril_dir)

    # Explicitly set the number of torch threads.
    if args["num_threads"] is not None:
        os.environ["OMP_NUM_THREADS"] = str(args["num_threads"])

    config = f_unpack_dict(config)
    if config.get("use_bias", False):
        if "bias_separation" in config and "addtl_bias_separation" in config and "output_scale" in config:
            # Do a hacky reconfigure.
            if config["output_scale"] > config["bias_separation"] + config["addtl_bias_separation"]:
                config["output_scale"] = config["bias_separation"] + config["addtl_bias_separation"]
        config["metric_loss_md"]["output_scale"] = config["output_scale"]

    output_dir = args["output_dir"]

    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    trial_dir = output_dir / f"embeddings_{dtime}_{os.getpid()}"
    trial_dir.mkdir(parents=True, exist_ok=False)

    # Seed
    seed = np.random.randint(1, 1e8)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config["seed"] = seed
    config["iterations_per_epoch"] = args["iterations_per_epoch"]

    logging.info(config)

    # Build trainer and train.
    trainer, epoch_end = build_trainer(
        config,
        f"{output_dir}/out.parquet",
        trial_dir,
        args["benchmark_config"],
        args["train_size"],
        dataloader_num_workers=0,
        disable_tqdm=True,
    )

    # Dump the config that we are executing.
    with open(f"{trial_dir}/config", "w") as f:
        f.write(json.dumps(config, indent=4))

    trainer.train(num_epochs=config["num_epochs"])
    if trainer.failed:
        # Trainer has failed.
        with open(f"{trial_dir}/FAILED", "w") as f:
            if trainer.fail_msg is not None:
                f.write(trainer.fail_msg)

        if trainer.fail_data is not None:
            torch.save(trainer.fail_data, f"{trial_dir}/fail_data.pth")
        session.report({"loss": 1e8})
    else:
        loss = epoch_end(trainer, force=True, suppress=True)["total_avg_loss"]
        session.report({"loss": loss})


def execute_train(args):
    # Set initial seed.
    seed = args.seed if args.seed != 0 else random.randint(0, 1e8)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = args.output_dir
    logging.getLogger().setLevel(logging.INFO)

    if not Path(f"{args.output_dir}/out.parquet").exists():
        # Produce table mapping.
        tbl_dirs = {}
        with open(args.benchmark_config, "r") as f:
            benchmark_config = yaml.safe_load(f)["mythril"]
            tables = benchmark_config["tables"]
            for i, tbl in enumerate(tables):
                tbl_dirs[tbl] = i

        files = []
        for comp in args.input_data.split(","):
            files.extend([f for f in Path(comp).rglob("*.parquet")])

        def read(file):
            tbl = Path(file).parts[-2]
            if tbl not in tbl_dirs:
                tbl = Path(file).parts[-3]
            df = pd.read_parquet(file)
            df["tbl_index"] = tbl_dirs[tbl]

            if args.pad_min is not None:
                if df.shape[0] < args.pad_min:
                    df = pd.concat([df] * int(args.pad_min / df.shape[0]))
            return df

        df = pd.concat(map(read, files))

        if "reference_cost" in df.columns:
            target_cost = df.target_cost

            # This expression is the improvement expression.
            act_cost = df.reference_cost - (df.table_reference_cost - target_cost)
            mult = (df.reference_cost / act_cost)
            rel = ((df.reference_cost - act_cost) / act_cost)
            mult_tbl = (df.table_reference_cost / target_cost)
            rel_tbl = ((df.table_reference_cost - target_cost) / target_cost)

            if args.table_shape:
                df["quant_mult_cost_improvement"] = quantile_transform(mult_tbl.values.reshape(-1, 1), n_quantiles=100000, subsample=df.shape[0])
                df["quant_rel_cost_improvement"] = quantile_transform(rel_tbl.values.reshape(-1, 1), n_quantiles=100000, subsample=df.shape[0])
            else:
                df["quant_mult_cost_improvement"] = quantile_transform(mult.values.reshape(-1, 1), n_quantiles=min(100000, df.shape[0]), subsample=df.shape[0])
                df["quant_rel_cost_improvement"] = quantile_transform(rel.values.reshape(-1, 1), n_quantiles=min(100000, df.shape[0]), subsample=df.shape[0])

            df.drop(columns=["reference_cost", "table_reference_cost", "target_cost"], inplace=True, errors="ignore")

        if args.inflate_ratio > 1:
            df = pd.concat([df] * args.inflate_ratio)

        if args.dual_class:
            df["real_idx_class"] = df["idx_class"]
            df["idx_class"] = df["real_idx_class"] * df.col0.max() + df.col1

        df.drop(columns=["table"], inplace=True)
        df.fillna(0, inplace=True)
        # Only int-ify non-cost columns.
        columns = [c for c in df.columns if c not in COST_COLUMNS and "idx_class" not in c and "cmd" != c]
        df[columns] = df[columns].astype(int)

        if args.rebias > 0:
            groups = df.groupby(by=["tbl_index", "idx_class"]).quant_mult_cost_improvement.describe().sort_values(by=["max"], ascending=False)
            datum = []
            cur_bias = 1.
            sep_bias = args.rebias
            for g in groups.itertuples():
                d = df[(df.tbl_index == g.Index[0]) & (df.idx_class == g.Index[1]) & (df.quant_mult_cost_improvement >= g._6)].copy()
                d["quant_mult_cost_improvement"] = cur_bias - (args.rebias / 2)
                datum.append(d)
                cur_bias -= sep_bias
            df = pd.concat(datum, ignore_index=True)

        df.to_parquet(f"{args.output_dir}/out.parquet")

    if args.gen_only:
        # We are done.
        return

    start_time = time.time()

    with open(args.config, "r") as f:
        json_dict = json.load(f)
        space = parse_hyperopt_config(json_dict["config"])

    if args.manual:
        execute_manual(args, space)
        return

    # Connect to cluster or die.
    ray.init(address="localhost:6379", log_to_driver=False)

    scheduler = FIFOScheduler()
    # Search.
    search = HyperOptSearch(
        metric="loss",
        mode="min",
        points_to_evaluate=None,
        n_initial_points=20,
        space=space,
    )
    search = ConcurrencyLimiter(search, max_concurrent=args.max_concurrent)
    tune_config = TuneConfig(
        scheduler=scheduler,
        search_alg=search,
        num_samples=args.num_trials,
        max_concurrent_trials=args.max_concurrent,
        chdir_to_trial_dir=True,
    )

    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_config = RunConfig(
        name=f"MythrilHPO_{dtime}",
        failure_config=FailureConfig(max_failures=0, fail_fast=True),
        sync_config=SyncConfig(upload_dir=None, syncer=None),
        verbose=2,
        log_to_file=True,
    )

    resources = {"cpu": 1} if args.ray_num_gpu == 0 else {"gpu": 1 / args.ray_num_gpu}
    trainable = with_resources(with_parameters(hpo_train, args=vars(args)), resources)

    # Hopefully this is now serializable.
    args = vars(args)
    args.pop("func")
    tuner = ray.tune.Tuner(
        trainable,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result(metric="loss", mode="min").config)
    if results.num_errors > 0:
        print("Encountered exceptions!")
        for i in range(len(results)):
            if results[i].error:
                print(f"Trial {results[i]} FAILED")
        assert False

    duration = time.time() - start_time
    with open(f"{output_dir}/hpo_train_time.txt", "w") as f:
        f.write(f"{duration}")
