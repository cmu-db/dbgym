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
import traceback
import logging
from datetime import datetime
from click.core import Context
from typing import Dict
import shutil
import itertools
import math

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

from misc.utils import open_and_save, restart_ray, DEFAULT_HPO_SPACE_RELPATH, default_benchmark_config_relpath, default_dataset_path, BENCHMARK_PLACEHOLDER, DATA_PATH_PLACEHOLDER


STATS_FNAME = "stats.txt"
RANGES_FNAME = "ranges.txt"


def _fetch_index_parameters(ctx: Context, benchmark: str, data: Dict):
    tables = data["protox"]["tables"]
    attributes = data["protox"]["attributes"]
    query_spec = data["protox"]["query_spec"]
    
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


def _load_input_data(ctx, input_path, train_size, max_attrs, require_cost, seed):
    # Load the input data.
    columns = []
    columns += ["tbl_index", "idx_class"]
    columns += [f"col{c}" for c in range(max_attrs - 1)]
    if require_cost:
        columns += COST_COLUMNS

    with open_and_save(ctx, input_path, mode="rb") as input_file:
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


def build_trainer(ctx, benchmark, config, input_path, trial_dir, benchmark_config_path, train_size, dataloader_num_workers=0, disable_tqdm=False):
    max_cat_features = 0
    max_attrs = 0

    # Load the benchmark configuration.
    with open_and_save(ctx, benchmark_config_path, "r") as f:
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
        input_path,
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


def hpo_train(config, ctx, generic_args, train_args):
    sys.path.append(os.fspath(ctx.obj.dbgym_repo_path))

    # Explicitly set the number of torch threads.
    os.environ["OMP_NUM_THREADS"] = str(train_args.train_max_concurrent)

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
    config["iterations_per_epoch"] = train_args.iterations_per_epoch

    logging.info(config)

    # Build trainer and train.
    trainer, epoch_end = build_trainer(
        ctx,
        generic_args.benchmark,
        config,
        generic_args.dataset_path,
        trial_dir,
        generic_args.benchmark_config_path,
        train_args.train_size,
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


class EmbeddingGenericArgs:
    '''Just used to reduce the # of parameters we pass into functions'''
    def __init__(self, benchmark, benchmark_config_path, dataset_path):
        self.benchmark = benchmark
        self.benchmark_config_path = benchmark_config_path
        self.dataset_path = dataset_path


class EmbeddingTrainArgs:
    '''Same comment as EmbeddingGenericArgs'''
    def __init__(self, hpo_space_path, train_max_concurrent, iterations_per_epoch, num_samples, train_size):
        self.hpo_space_path = hpo_space_path
        self.train_max_concurrent = train_max_concurrent
        self.iterations_per_epoch = iterations_per_epoch
        self.num_samples = num_samples
        self.train_size = train_size


class EmbeddingAnalyzeArgs:
    '''Same comment as EmbeddingGenericArgs'''
    def __init__(self, start_epoch, batch_size, num_batches, max_segments, num_points_to_sample, num_classes_to_keep):
        self.start_epoch = start_epoch
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.max_segments = max_segments
        self.num_points_to_sample = num_points_to_sample
        self.num_classes_to_keep = num_classes_to_keep


# click setup
@click.command()
@click.pass_context

# generic args
@click.argument("benchmark")
@click.option("--benchmark-config-path", default=None, type=str, help=f"The path to the .yaml config file for the benchmark. The default is {default_benchmark_config_relpath(BENCHMARK_PLACEHOLDER)}.")
@click.option("--dataset-path", default=None, type=str, help=f"The path to the .parquet file containing the training data to use to train the embedding models. The default is {default_dataset_path(DATA_PATH_PLACEHOLDER, BENCHMARK_PLACEHOLDER)}.")

# train args
@click.option("--hpo-space-path", default=DEFAULT_HPO_SPACE_RELPATH, type=str, help="The path to the .json file defining the search space for hyperparameter optimization (HPO).")
@click.option("--train-max-concurrent", default=1, type=int, help="The max # of concurrent embedding models to train. Setting this too high may overload the machine.")
@click.option("--iterations-per-epoch", default=1000, help=f"TODO(wz2)")
@click.option("--num-samples", default=40, help=f"The # of times to specific hyperparameter configs to sample from the hyperparameter search space and train embedding models with.")
@click.option("--train-size", default=0.99, help=f"TODO(wz2)")

# analyze args
@click.option("--start-epoch", default=0, help="The epoch to start analyzing models at.")
@click.option("--batch-size", default=8192, help=f"The size of batches to use to build {STATS_FNAME}.")
@click.option("--num-batches", default=100, help=f"The number of batches to use to build {STATS_FNAME}. Setting it to -1 indicates \"use all batches\".")
@click.option("--max-segments", default=15, help=f"The maximum # of segments in the latent space when creating {RANGES_FNAME}.")
@click.option("--num-points-to-sample", default=8192, help=f"The number of points to sample when creating {RANGES_FNAME}.")
@click.option("--num-classes-to-keep", default=5, help=f"The number of classes to keep for each segment when creating {RANGES_FNAME}.")

# misc args
@click.option("--seed", default=None, type=int, help="The seed used for all sources of randomness (random, np, torch, etc.). The default is a random value.")

def train(ctx, benchmark, benchmark_config_path, dataset_path, hpo_space_path, train_max_concurrent, iterations_per_epoch, num_samples, train_size, start_epoch, batch_size, num_batches, max_segments, num_points_to_sample, num_classes_to_keep, seed):
    '''
    Trains embeddings based on num_samples samples of the hyperparameter space. Analyzes the accuracy of all epochs of all hyperparameter
    space samples. Selects the best embedding and packages it as a .pth file in the run_*/ dir.
    '''
    # set args to defaults programmatically (do this BEFORE creating arg objects)
    if dataset_path == None:
        dataset_path = default_dataset_path(ctx.obj.dbgym_data_path, benchmark)
    # TODO(phw2): figure out whether different scale factors use the same config
    # TODO(phw2): figure out what parts of the config should be taken out (like stuff about tables)
    if benchmark_config_path == None:
        benchmark_config_path = default_benchmark_config_relpath(benchmark)
    if seed == None:
        seed = random.randint(0, 1e8)

    # setup
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logging.getLogger().setLevel(logging.INFO)

    # group args together to reduce the # of parameters we pass into functions
    # I chose to group them into separate objects instead because it felt hacky to pass a giant args object into every function
    # I didn't group misc args because they're just miscellaneous
    generic_args = EmbeddingGenericArgs(benchmark, benchmark_config_path, dataset_path)
    train_args = EmbeddingTrainArgs(hpo_space_path, train_max_concurrent, iterations_per_epoch, num_samples, train_size)
    analyze_args = EmbeddingAnalyzeArgs(start_epoch, batch_size, num_batches, max_segments, num_points_to_sample, num_classes_to_keep)

    # run all steps of training
    train_all_models(ctx, generic_args, train_args)
    num_parts = compute_num_parts(num_samples)
    redist_trained_models(ctx, num_parts)
    analyze_all_embeddings_parts(ctx, num_parts, generic_args, analyze_args)


def train_all_models(ctx, generic_args, train_args):
    '''
    Trains all num_samples models using different samples of the hyperparameter space, writing their
    results to different embedding_*/ folders in the run_*/ folder
    '''
    start_time = time.time()

    with open_and_save(ctx, train_args.hpo_space_path, "r") as f:
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
    search = ConcurrencyLimiter(search, max_concurrent=train_args.train_max_concurrent)
    tune_config = TuneConfig(
        scheduler=scheduler,
        search_alg=search,
        num_samples=train_args.num_samples,
        max_concurrent_trials=train_args.train_max_concurrent,
        chdir_to_trial_dir=True,
    )

    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_config = RunConfig(
        name=f"ProtoXEmbeddingHPO_{dtime}",
        storage_path=None,
        failure_config=FailureConfig(max_failures=0, fail_fast=True),
        sync_config=SyncConfig(),
        verbose=2,
        log_to_file=True,
    )

    resources = {"cpu": 1}
    trainable = with_resources(with_parameters(hpo_train, ctx=ctx, generic_args=generic_args, train_args=train_args), resources)

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
                print(traceback.print_exception(results[i].error))
        assert False

    duration = time.time() - start_time
    with open(f"{ctx.obj.dbgym_this_run_path}/hpo_train_time.txt", "w") as f:
        f.write(f"{duration}")


def compute_num_parts(num_samples):
    # TODO(phw2): in the future, implement running different parts in parallel, set OMP_NUM_THREADS accordingly, and investigate the effect of having more parts
    # TODO(phw2): if having more parts is effective, figure out a good way to specify num_parts (can it be determined automatically or should it be a CLI arg?)
    # TODO(phw2): does anything bad happen if num_parts doesn't evenly divide num_samples?
    return 1


def get_part_i_dpath(ctx, part_i) -> str:
    return os.path.join(ctx.obj.dbgym_this_run_path, f"part{part_i}")


# TODO(phw2): check if anything goes wrong in any part of the entire selection process if num_parts doesn't evenly divide num_samples
def redist_trained_models(ctx, num_parts):
    '''
    Redistribute all embeddings_*/ folders inside the run_*/ folder into num_parts subfolders
    '''
    inputs = [f for f in ctx.obj.dbgym_this_run_path.glob("embeddings*") if os.path.isdir(f)]

    for part_i in range(num_parts):
        Path(get_part_i_dpath(ctx, part_i)).mkdir(parents=True, exist_ok=True)

    for model_i, emb in enumerate(inputs):
        part_i = model_i % num_parts
        shutil.move(emb, get_part_i_dpath(ctx, part_i))


def analyze_all_embeddings_parts(ctx, num_parts, generic_args, analyze_args):
    '''
    Analyze all part*/ dirs _in parallel_
    '''
    # TODO(phw2): figure out how to do this in parallel
    for part_i in range(num_parts):
        analyze_embeddings_part(ctx, part_i, generic_args, analyze_args)


def analyze_embeddings_part(ctx, part_i, generic_args, analyze_args):
    '''
    Analyze (meaning create both stats.txt and ranges.txt) all the embedding models in the part[part_i]/ dir
    '''
    part_dpath = get_part_i_dpath(ctx, part_i)

    start_time = time.time()
    create_stats_for_part(ctx, part_dpath, generic_args, analyze_args)
    duration = time.time() - start_time
    with open(os.path.join(part_dpath, "stats_time.txt"), "w") as f:
        f.write(f"{duration}")

    start_time = time.time()
    create_ranges_for_part(ctx, part_dpath, generic_args, analyze_args)
    duration = time.time() - start_time
    with open(os.path.join(part_dpath, "ranges_time.txt"), "w") as f:
        f.write(f"{duration}")


def create_stats_for_part(ctx, part_dpath, generic_args, analyze_args):
    '''
    Creates a stats.txt file inside each embeddings_*/models/epoch*/ dir inside this part*/ dir
    TODO(wz2): what does stats.txt contain?
    '''
    # Unlike for training, we're safe to use all threads for creating stats
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

    # Load the benchmark configuration.
    with open_and_save(ctx, generic_args.benchmark_config_path, "r") as f:
        data = yaml.safe_load(f)
        max_attrs, max_cat_features, _, _ = _fetch_index_parameters(ctx, generic_args.benchmark, data)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    models = itertools.chain(*[Path(part_dpath).rglob("config")])
    models = [m for m in models]
    for model_config in tqdm.tqdm(models):
        if ((Path(model_config).parent) / "FAILED").exists():
            print("Detected failure in: ", model_config)
            continue

        with open_and_save(ctx, model_config, "r") as f:
            config = json.load(f)

        # Create them here since these are constant for a given "model" configuration.
        dataset, idx_class, num_classes = None, None, None
        class_mapping = None
        metric_loss_fn, vae_loss = None, None
        vae = _create_vae_model(config, max_attrs, max_cat_features)
        require_cost = config["metric_loss_md"].get("require_cost", False)

        submodules = [f for f in (Path(model_config).parent / "models").glob("*")]
        submodules = sorted(submodules, key=lambda x: int(str(x).split("epoch")[-1]))
        # This is done for semantic sense since the "first" is actually at no epoch.
        modules = [submodules[r] for r in range(-1, len(submodules)) if r >= 0]
        if modules[0] != submodules[0]:
            modules = [submodules[0]] + modules

        if modules[-1] != submodules[-1]:
            modules.append(submodules[-1])

        modules = [m for m in modules if int(str(m).split("epoch")[-1]) >= analyze_args.start_epoch]

        for i, module in tqdm.tqdm(enumerate(modules), total=len(modules), leave=False):
            epoch = int(str(module).split("epoch")[-1])
            module_path = os.path.join(module, f"embedder_{epoch}.pth")

            if Path(os.path.join(module, f"{STATS_FNAME}")).exists():
                continue

            # Load the specific epoch model.
            vae.load_state_dict(torch.load(module_path, map_location=device))
            vae.to(device=device).eval()
            collate_fn = gen_vae_collate(max_cat_features)

            if dataset is None:
                # Get the dataset if we need to.
                dataset, _, idx_class, _, num_classes = _load_input_data(
                    ctx,
                    generic_args.dataset_path,
                    1.,
                    max_attrs,
                    require_cost,
                    seed=0)

                class_mapping = []
                for c in range(num_classes):
                    if idx_class[idx_class == c].shape[0] > 0:
                        class_mapping.append(c)

                # Use a common loss function.
                metric_loss_fn = CostLoss(config["metric_loss_md"])
                vae_loss = VAELoss(config["loss_fn"], max_attrs, max_cat_features)

            # Construct the accumulator.
            accumulated_stats = {}
            for class_idx in class_mapping:
                accumulated_stats[f"recon_{class_idx}"] = []

            analyze_all_batches = analyze_args.num_batches == -1
            if analyze_args.num_batches > 0 or analyze_all_batches:
                accumulated_stats.update({
                    "recon_accum": [],
                    "metric_accum": [],
                })

                # Setup the dataloader.
                if analyze_all_batches:
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=analyze_args.batch_size, collate_fn=collate_fn)
                    total = len(dataloader)
                else:
                    sampler = StratifiedRandomSampler(idx_class, max_class=num_classes, batch_size=analyze_args.batch_size, allow_repeats=False)
                    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=analyze_args.batch_size, collate_fn=collate_fn)
                    total = min(len(sampler), analyze_args.num_batches)
                error = False
                with torch.no_grad():
                    with tqdm.tqdm(total=total, leave=False) as pbar:
                        for (x, y) in dataloader:
                            x = x.to(device=device)

                            if config["use_bias"]:
                                bias_fn = get_bias_fn(config)
                                bias = bias_fn(x, y)
                                if isinstance(bias, torch.Tensor):
                                    bias = bias.to(device=device)
                                else:
                                    lbias = bias[0].to(device=device)
                                    hbias = bias[1].to(device=device)
                                    bias = (lbias, hbias)
                            else:
                                bias = None

                            # Pass it through the VAE with the settings.
                            z, decoded, error = vae(x, bias=bias)
                            if error:
                                # If we've encountered an error, abort early.
                                # Don't use a model that can produce errors.
                                break

                            # Flatten.
                            classes = y[:, -1].flatten()

                            assert metric_loss_fn is not None
                            loss_dict = vae_loss.compute_loss(
                                preds=decoded,
                                unused0=None,
                                unused1=None,
                                data=(x, y),
                                is_eval=True)

                            assert vae_loss.loss_fn is not None
                            for class_idx in class_mapping:
                                y_mask = classes == class_idx
                                x_extract = x[y_mask.bool()]
                                if x_extract.shape[0] > 0:
                                    decoded_extract = decoded[y_mask.bool()]
                                    loss = vae_loss.loss_fn(decoded_extract, x_extract, y[y_mask.bool()])
                                    accumulated_stats[f"recon_{class_idx}"].append(loss.mean().item())

                            input_y = y
                            if y.shape[1] == 1:
                                input_y = y.flatten()

                            metric_loss = metric_loss_fn(z, input_y, None).item()
                            accumulated_stats["recon_accum"].append(loss_dict["recon_loss"]["losses"].item())
                            accumulated_stats["metric_accum"].append(metric_loss)

                            del z
                            del x
                            del y

                            # Break out if we are done.
                            pbar.update(1)
                            total -= 1
                            if total == 0:
                                break

                # Output the evaluated stats.
                with open(os.path.join(module, f"{STATS_FNAME}"), "w") as f:
                    stats = {
                        stat_key: (stats if isinstance(stats, np.ScalarType) else (np.mean(stats) if len(stats) > 0 else 0))
                        for stat_key, stats in accumulated_stats.items()
                    }
                    stats["error"] = error.item()
                    f.write(json.dumps(stats, indent=4))

                del dataloader
                gc.collect()
                gc.collect()


def create_ranges_for_part(ctx, part_dpath, generic_args, analyze_args):
    '''
    Create the ranges.txt for all models in part_dpath
    TODO(wz2): what does ranges.txt contain?
    '''
    # Unlike for training, we're safe to use all threads for creating ranges
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    paths = sorted([f for f in Path(part_dpath).rglob("embedder_*.pth") if "optimizer" not in str(f)])
    for embedder_fpath in tqdm.tqdm(paths):
        create_ranges_for_embedder(ctx, embedder_fpath, generic_args, analyze_args)


def create_ranges_for_embedder(ctx, embedder_fpath, generic_args, analyze_args):
    '''
    Create the ranges.txt file corresponding to a specific part*/embeddings_*/models/epoch*/embedder_*.pth file
    '''
    # Return right away if the epoch isn't high enough
    epoch_i = int(str(embedder_fpath).split("embedder_")[-1].split(".pth")[0])
    if epoch_i < analyze_args.start_epoch:
        return
    
    # Load the benchmark configuration.
    with open_and_save(ctx, generic_args.benchmark_config_path, "r") as f:
        data = yaml.safe_load(f)
        tables = data["protox"]["tables"]
        max_attrs, max_cat_features, att_usage, class_mapping = _fetch_index_parameters(ctx, generic_args.benchmark, data)

    # don't use open_and_save() because we generated embeddings_config_fpath in this run
    embeddings_dpath = embedder_fpath.parent.parent.parent # part*/embeddings_*/
    embeddings_config_fpath = os.path.join(embeddings_dpath, "config") # part*/embeddings_*/config
    with open(embeddings_config_fpath, "r") as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = _create_vae_model(config, max_attrs, max_cat_features)
    # Load the specific epoch model.
    vae.load_state_dict(torch.load(embedder_fpath, map_location=device))
    vae.to(device=device).eval()

    idxs = IndexSpace(
        agent_type="wolp",
        tables=tables,
        max_num_columns=0,
        index_repr=IndexRepr.ONE_HOT_DETERMINISTIC.name,
        seed=np.random.randint(1, 1e10),
        latent_dim=config["latent_dim"],
        index_vae_model=vae,
        index_output_scale=1.,
        attributes_overwrite=att_usage)
    idxs.rel_metadata = att_usage
    idxs._build_mapping(att_usage)

    def decode_to_classes(rand_points):
        with torch.no_grad():
            rand_decoded = idxs._decode(act=rand_points)
            classes = {}
            for r in range(rand_points.shape[0]):
                act = idxs.index_repr_policy.sample_action(idxs.np_random, rand_decoded[r], att_usage, False, True)
                idx_class = idxs.get_index_class(act)
                if idx_class not in classes:
                    classes[idx_class] = 0
                classes[idx_class] += 1
        return sorted([(k, v) for k, v in classes.items()], key=lambda x: x[1], reverse=True)

    output_scale = config["metric_loss_md"]["output_scale"]
    bias_separation = config["metric_loss_md"]["bias_separation"]
    num_segments = min(analyze_args.max_segments, math.ceil(1.0 / bias_separation))

    base = 0
    epoch_dpath = os.path.join(embeddings_dpath, "models", f"epoch{epoch_i}") # part*/embeddings_*/models/epoch*/
    ranges_fpath = os.path.join(epoch_dpath, RANGES_FNAME)
    with open(ranges_fpath, "w") as f:
        for _ in tqdm.tqdm(range(num_segments), total=num_segments, leave=False):
            classes = decode_to_classes(torch.rand(analyze_args.num_points_to_sample, config["latent_dim"]) * output_scale + base)
            if analyze_args.num_classes_to_keep != 0:
                classes = classes[:analyze_args.num_classes_to_keep]

            f.write(f"Generating range {base} - {base + output_scale}\n")
            f.write("\n".join([f"{k}: {v / analyze_args.num_points_to_sample}" for (k, v) in classes]))
            f.write("\n")
            base += output_scale