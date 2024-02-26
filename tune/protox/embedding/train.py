import sys
import os
import yaml
import json
import random
import tqdm
import click
import numpy as np
import logging
import gc
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import time
import logging
from datetime import datetime
from click.core import Context
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from pytorch_metric_learning.utils import logging_presets

import ray
from ray.train import FailureConfig, RunConfig, SyncConfig
from ray.tune import with_resources, with_parameters, TuneConfig
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
from ray.air import session

from tune.protox.embedding.loss import COST_COLUMNS, CostLoss, get_bias_fn
from tune.protox.embedding.vae import gen_vae_collate, VAE, VAELoss
from tune.protox.embedding.trainer import VAETrainer, StratifiedRandomSampler
from tune.protox.embedding.utils import f_unpack_dict, parse_hyperopt_config

from tune.protox.env.workload import Workload
from tune.protox.env.space.index_space import IndexSpace
from tune.protox.env.space.index_policy import IndexRepr

from misc.utils import open_and_save, DEFAULT_HPO_SPACE_RELPATH, default_benchmark_config_relpath, restart_ray


def _fetch_index_parameters(ctx: Context, benchmark: str, data: Dict):
    tables = data["dbgym"]["tables"]
    attributes = data["dbgym"]["attributes"]
    query_spec = data["dbgym"]["query_spec"]
    
    # TODO(phw2): figure out how to pass query_directory. should it in the .yaml or should it be a CLI args?
    if "query_directory" not in query_spec:
        assert "query_order" not in query_spec
        query_spec["query_directory"] = os.path.join(ctx.obj.dbgym_data_path, f'{benchmark}_queries')
        query_spec["query_order"] = os.path.join(query_spec["query_directory"], f'order.txt')

    workload = Workload(ctx, tables, attributes, query_spec, pid=None)
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


def _load_input_data(ctx, input_fpath, train_size, max_attrs, require_cost, seed):
    # Load the input data.
    columns = []
    columns += ["tbl_index", "idx_class"]
    columns += [f"col{c}" for c in range(max_attrs - 1)]
    if require_cost:
        columns += COST_COLUMNS

    with open_and_save(ctx, input_fpath, mode="rb") as input_file:
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


def build_trainer(ctx, benchmark, config, input_fpath, trial_dir, benchmark_config_fpath, train_size, dataloader_num_workers=0, disable_tqdm=False):
    max_cat_features = 0
    max_attrs = 0

    # Load the benchmark configuration.
    with open_and_save(ctx, benchmark_config_fpath, "r") as f:
        data = yaml.safe_load(f)
        max_attrs, max_cat_features, att_usage, class_mapping = _fetch_index_parameters(ctx, benchmark, data)

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
        ctx,
        input_fpath,
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


def hpo_train(config, ctx, benchmark, max_concurrent, iterations_per_epoch, benchmark_config_fpath, train_size):
    sys.path.append(os.fspath(ctx.obj.dbgym_repo_path))

    # Explicitly set the number of torch threads.
    os.environ["OMP_NUM_THREADS"] = str(max_concurrent)

    config = f_unpack_dict(config)
    if config.get("use_bias", False):
        if "bias_separation" in config and "addtl_bias_separation" in config and "output_scale" in config:
            # Do a hacky reconfigure.
            if config["output_scale"] > config["bias_separation"] + config["addtl_bias_separation"]:
                config["output_scale"] = config["bias_separation"] + config["addtl_bias_separation"]
        config["metric_loss_md"]["output_scale"] = config["output_scale"]

    output_dir = ctx.obj.dbgym_this_run_path

    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    trial_dir = output_dir / f"embeddings_{dtime}_{os.getpid()}"
    trial_dir.mkdir(parents=True, exist_ok=False)

    # Seed
    seed = np.random.randint(1, 1e8)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config["seed"] = seed
    config["iterations_per_epoch"] = iterations_per_epoch

    logging.info(config)

    # Build trainer and train.
    trainer, epoch_end = build_trainer(
        ctx,
        benchmark,
        config,
        os.path.join(ctx.obj.dbgym_data_path, f'{benchmark}_embedding_traindata.parquet'),
        trial_dir,
        benchmark_config_fpath,
        train_size,
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


@click.command()
@click.option("--seed", default=None, type=int, help="The seed used for all sources of randomness (random, np, torch, etc.). The default is a random value.")
@click.option("--max-concurrent", default=1, type=int, help="The max # of concurrent embedding models to train. Setting this too high may overload the machine.")
@click.option("--hpo-space-fpath", default=DEFAULT_HPO_SPACE_RELPATH, type=str, help="The path to the .json file defining the search space for hyperparameter optimization (HPO).")
@click.option("--benchmark-config-fpath", default=None, type=str, help=f"The path to the .yaml config file for the benchmark. The default is {default_benchmark_config_relpath('[benchmark]')}")
@click.option("--iterations-per-epoch", default=1000, help=f"TODO(wz2)")
@click.option("--num-samples", default=40, help=f"The # of times to specific hyperparameter configs to sample from the hyperparameter search space and train an embedding model with.")
@click.option("--train-size", default=0.99, help=f"TODO(wz2)")
@click.argument("benchmark")
@click.pass_context
def train(ctx, benchmark, seed, max_concurrent, hpo_space_fpath, benchmark_config_fpath, iterations_per_epoch, num_samples, train_size):
    # set args to defaults programmatically
    if seed == None:
        seed = random.randint(0, 1e8)
    # TODO(phw2): figure out whether different scale factors use the same config
    # TODO(phw2): figure out what parts of the config should be taken out (like stuff about tables)
    if benchmark_config_fpath == None:
        benchmark_config_fpath = default_benchmark_config_relpath(benchmark)

    # set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logging.getLogger().setLevel(logging.INFO)

    start_time = time.time()

    with open_and_save(ctx, hpo_space_fpath, "r") as f:
        json_dict = json.load(f)
        space = parse_hyperopt_config(json_dict["config"])

    # Connect to cluster or die.
    restart_ray()
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
    search = ConcurrencyLimiter(search, max_concurrent=max_concurrent)
    tune_config = TuneConfig(
        scheduler=scheduler,
        search_alg=search,
        num_samples=num_samples,
        max_concurrent_trials=max_concurrent,
        chdir_to_trial_dir=True,
    )

    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_config = RunConfig(
        name=f"MythrilHPO_{dtime}",
        storage_path=None,
        failure_config=FailureConfig(max_failures=0, fail_fast=True),
        sync_config=SyncConfig(),
        verbose=2,
        log_to_file=True,
    )

    resources = {"cpu": 1}
    trainable = with_resources(with_parameters(hpo_train, ctx=ctx, benchmark=benchmark, max_concurrent=max_concurrent, iterations_per_epoch=iterations_per_epoch, benchmark_config_fpath=benchmark_config_fpath, train_size=train_size), resources)

    # Hopefully this is now serializable.
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0" # makes it so Ray doesn't change dir
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
    with open(f"{ctx.obj.dbgym_this_run_path}/hpo_train_time.txt", "w") as f:
        f.write(f"{duration}")