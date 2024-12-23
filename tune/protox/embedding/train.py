import logging
import random
from pathlib import Path
from typing import Optional

import click
import numpy as np
import torch

from benchmark.constants import DEFAULT_SCALE_FACTOR
from tune.protox.embedding.analyze import (
    RANGES_FNAME,
    STATS_FNAME,
    analyze_all_embeddings_parts,
    compute_num_parts,
    redist_trained_models,
)
from tune.protox.embedding.select import select_best_embeddings
from tune.protox.embedding.train_all import train_all_embeddings
from tune.protox.embedding.train_args import (
    EmbeddingAnalyzeArgs,
    EmbeddingSelectArgs,
    EmbeddingTrainAllArgs,
    EmbeddingTrainGenericArgs,
)
from util.workspace import (
    BENCHMARK_NAME_PLACEHOLDER,
    DEFAULT_HPO_SPACE_PATH,
    WORKLOAD_NAME_PLACEHOLDER,
    WORKSPACE_PATH_PLACEHOLDER,
    DBGymConfig,
    default_benchmark_config_path,
    default_traindata_path,
    default_workload_path,
    fully_resolve_inputpath,
    get_default_workload_name_suffix,
    get_workload_name,
)


# click setup
@click.command()
@click.pass_obj

# generic args
@click.argument("benchmark-name", type=str)
@click.option(
    "--workload-name-suffix",
    type=str,
    default=None,
    help=f"The suffix of the workload name (the part after the scale factor).",
)
@click.option(
    "--scale-factor",
    type=float,
    default=DEFAULT_SCALE_FACTOR,
    help=f"The scale factor used when generating the data of the benchmark.",
)
@click.option(
    "--benchmark-config-path",
    type=Path,
    default=None,
    help=f"The path to the .yaml config file for the benchmark. The default is {default_benchmark_config_path(BENCHMARK_NAME_PLACEHOLDER)}.",
)
@click.option(
    "--traindata-path",
    type=Path,
    default=None,
    help=f"The path to the .parquet file containing the training data to use to train the embedding models. The default is {default_traindata_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER)}.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="The seed used for all sources of randomness (random, np, torch, etc.). The default is a random value.",
)

# train args
@click.option(
    "--hpo-space-path",
    type=Path,
    default=DEFAULT_HPO_SPACE_PATH,
    help="The path to the .json file defining the search space for hyperparameter optimization (HPO).",
)
@click.option(
    "--train-max-concurrent",
    type=int,
    default=1,
    help="The max # of concurrent embedding models to train during hyperparameter optimization. This is usually set lower than `nproc` to reduce memory pressure.",
)
@click.option("--iterations-per-epoch", default=1000, help=f"TODO(wz2)")
@click.option(
    "--num-samples",
    type=int,
    default=40,
    help=f"The # of times to specific hyperparameter configs to sample from the hyperparameter search space and train embedding models with.",
)
@click.option("--train-size", type=float, default=0.99, help=f"TODO(wz2)")

# analyze args
@click.option(
    "--start-epoch", type=int, default=0, help="The epoch to start analyzing models at."
)
@click.option(
    "--batch-size",
    type=int,
    default=8192,
    help=f"The size of batches to use to build {STATS_FNAME}.",
)
@click.option(
    "--num-batches",
    type=int,
    default=100,
    help=f'The number of batches to use to build {STATS_FNAME}. Setting it to -1 indicates "use all batches".',
)
@click.option(
    "--max-segments",
    type=int,
    default=15,
    help=f"The maximum # of segments in the latent space when creating {RANGES_FNAME}.",
)
@click.option(
    "--num-points-to-sample",
    type=int,
    default=8192,
    help=f"The number of points to sample when creating {RANGES_FNAME}.",
)
@click.option(
    "--num-classes-to-keep",
    type=int,
    default=5,
    help=f"The number of classes to keep for each segment when creating {RANGES_FNAME}.",
)

# select args
@click.option(
    "--recon",
    type=float,
    default=None,
    help="The reconstruction error threshold our selected model(s) need to pass.",
)
@click.option(
    "--latent-dim",
    type=int,
    default=None,
    help="The # of latent dimensions our selected model(s) need to have.",
)
@click.option(
    "--bias-sep",
    type=float,
    default=None,
    help="The bias separation our selected model(s) need to have.",
)
@click.option(
    "--idx-limit",
    type=int,
    default=15,
    help="The number of indexes whose errors to compute during _attach().",
)
@click.option(
    "--num-curate", type=int, default=1, help="The number of models to curate"
)  # TODO(wz2): why would we want to curate more than one?
@click.option(
    "--allow-all", is_flag=True, help="Whether to curate within or across parts."
)
@click.option("--flatten-idx", type=int, default=0, help="TODO(wz2)")
def train(
    dbgym_cfg: DBGymConfig,
    benchmark_name: str,
    workload_name_suffix: str,
    scale_factor: float,
    benchmark_config_path: Optional[Path],
    traindata_path: Optional[Path],
    seed: Optional[int],
    hpo_space_path: Path,
    train_max_concurrent: int,
    iterations_per_epoch: int,
    num_samples: int,
    train_size: float,
    start_epoch: int,
    batch_size: int,
    num_batches: int,
    max_segments: int,
    num_points_to_sample: int,
    num_classes_to_keep: int,
    recon: float,
    latent_dim: int,
    bias_sep: float,
    idx_limit: int,
    num_curate: int,
    allow_all: bool,
    flatten_idx: int,
) -> None:
    """
    Trains embeddings with num_samples samples of the hyperparameter space.
    Analyzes the accuracy of all epochs of all hyperparameter space samples.
    Selects the best embedding(s) and packages it as a .pth file in the run_*/ dir.
    """
    # set args to defaults programmatically (do this before doing anything else in the function)
    if workload_name_suffix is None:
        workload_name_suffix = get_default_workload_name_suffix(benchmark_name)
    workload_name = get_workload_name(scale_factor, workload_name_suffix)
    if traindata_path is None:
        traindata_path = default_traindata_path(
            dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name
        )
    # TODO(phw2): figure out whether different scale factors use the same config
    # TODO(phw2): figure out what parts of the config should be taken out (like stuff about tables)
    if benchmark_config_path is None:
        benchmark_config_path = default_benchmark_config_path(benchmark_name)
    if seed is None:
        seed = random.randint(0, int(1e8))

    # Fully resolve all input paths.
    benchmark_config_path = fully_resolve_inputpath(dbgym_cfg, benchmark_config_path)
    traindata_path = fully_resolve_inputpath(dbgym_cfg, traindata_path)
    hpo_space_path = fully_resolve_inputpath(dbgym_cfg, hpo_space_path)

    # setup
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    workload_path = fully_resolve_inputpath(
        dbgym_cfg,
        default_workload_path(
            dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name
        ),
    )
    # group args. see comment in datagen.py:datagen()
    generic_args = EmbeddingTrainGenericArgs(
        benchmark_name,
        workload_name,
        scale_factor,
        benchmark_config_path,
        traindata_path,
        seed,
        workload_path,
    )
    train_args = EmbeddingTrainAllArgs(
        hpo_space_path,
        train_max_concurrent,
        iterations_per_epoch,
        num_samples,
        train_size,
    )
    analyze_args = EmbeddingAnalyzeArgs(
        start_epoch,
        batch_size,
        num_batches,
        max_segments,
        num_points_to_sample,
        num_classes_to_keep,
    )
    select_args = EmbeddingSelectArgs(
        recon, latent_dim, bias_sep, idx_limit, num_curate, allow_all, flatten_idx
    )

    # run all steps
    train_all_embeddings(dbgym_cfg, generic_args, train_args)
    num_parts = compute_num_parts(num_samples)
    redist_trained_models(dbgym_cfg, num_parts)
    analyze_all_embeddings_parts(dbgym_cfg, num_parts, generic_args, analyze_args)
    select_best_embeddings(dbgym_cfg, generic_args, select_args)
