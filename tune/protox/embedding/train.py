import logging
import random
from pathlib import Path

import click
import numpy as np
import torch

from misc.utils import (
    BENCHMARK_NAME_PLACEHOLDER,
    DEFAULT_HPO_SPACE_RELPATH,
    WORKLOAD_NAME_PLACEHOLDER,
    WORKSPACE_PATH_PLACEHOLDER,
    SCALE_FACTOR_PLACEHOLDER,
    conv_inputpath_to_abspath,
    default_benchmark_config_relpath,
    default_dataset_path,
    default_workload_path,
)
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


# click setup
@click.command()
@click.pass_obj

# generic args
@click.argument("benchmark-name", type=str)
@click.argument("workload-name", type=str)
@click.option(
    "--scale-factor",
    default=1.0,
    help=f"The scale factor used when generating the data of the benchmark.",
)
@click.option(
    "--benchmark-config-path",
    default=None,
    type=Path,
    help=f"The path to the .yaml config file for the benchmark. The default is {default_benchmark_config_relpath(BENCHMARK_NAME_PLACEHOLDER)}.",
)
@click.option(
    "--dataset-path",
    default=None,
    type=Path,
    help=f"The path to the .parquet file containing the training data to use to train the embedding models. The default is {default_dataset_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER, SCALE_FACTOR_PLACEHOLDER)}.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="The seed used for all sources of randomness (random, np, torch, etc.). The default is a random value.",
)

# train args
@click.option(
    "--hpo-space-path",
    default=DEFAULT_HPO_SPACE_RELPATH,
    type=str,
    help="The path to the .json file defining the search space for hyperparameter optimization (HPO).",
)
@click.option(
    "--train-max-concurrent",
    default=1,
    type=int,
    help="The max # of concurrent embedding models to train during hyperparameter optimization. This is usually set lower than `nproc` to reduce memory pressure.",
)
@click.option("--iterations-per-epoch", default=1000, help=f"TODO(wz2)")
@click.option(
    "--num-samples",
    default=40,
    help=f"The # of times to specific hyperparameter configs to sample from the hyperparameter search space and train embedding models with.",
)
@click.option("--train-size", default=0.99, help=f"TODO(wz2)")

# analyze args
@click.option(
    "--start-epoch", default=0, help="The epoch to start analyzing models at."
)
@click.option(
    "--batch-size",
    default=8192,
    help=f"The size of batches to use to build {STATS_FNAME}.",
)
@click.option(
    "--num-batches",
    default=100,
    help=f'The number of batches to use to build {STATS_FNAME}. Setting it to -1 indicates "use all batches".',
)
@click.option(
    "--max-segments",
    default=15,
    help=f"The maximum # of segments in the latent space when creating {RANGES_FNAME}.",
)
@click.option(
    "--num-points-to-sample",
    default=8192,
    help=f"The number of points to sample when creating {RANGES_FNAME}.",
)
@click.option(
    "--num-classes-to-keep",
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
    "--num-curate", default=1, help="The number of models to curate"
)  # TODO(wz2): why would we want to curate more than one?
@click.option(
    "--allow-all", is_flag=True, help="Whether to curate within or across parts."
)
@click.option("--flatten-idx", default=0, help="TODO(wz2)")
def train(
    dbgym_cfg,
    benchmark_name,
    workload_name,
    scale_factor,
    benchmark_config_path,
    dataset_path,
    seed,
    hpo_space_path,
    train_max_concurrent,
    iterations_per_epoch,
    num_samples,
    train_size,
    start_epoch,
    batch_size,
    num_batches,
    max_segments,
    num_points_to_sample,
    num_classes_to_keep,
    recon,
    latent_dim,
    bias_sep,
    idx_limit,
    num_curate,
    allow_all,
    flatten_idx,
):
    """
    Trains embeddings with num_samples samples of the hyperparameter space.
    Analyzes the accuracy of all epochs of all hyperparameter space samples.
    Selects the best embedding(s) and packages it as a .pth file in the run_*/ dir.
    """
    # set args to defaults programmatically (do this before doing anything else in the function)
    if dataset_path == None:
        dataset_path = default_dataset_path(
            dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name, scale_factor
        )
    # TODO(phw2): figure out whether different scale factors use the same config
    # TODO(phw2): figure out what parts of the config should be taken out (like stuff about tables)
    if benchmark_config_path == None:
        benchmark_config_path = default_benchmark_config_relpath(benchmark_name)
    if seed == None:
        seed = random.randint(0, 1e8)

    # Convert all input paths to absolute paths
    benchmark_config_path = conv_inputpath_to_abspath(dbgym_cfg, benchmark_config_path)
    dataset_path = conv_inputpath_to_abspath(dbgym_cfg, dataset_path)
    hpo_space_path = conv_inputpath_to_abspath(dbgym_cfg, hpo_space_path)

    # setup
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logging.getLogger().setLevel(logging.INFO)

    workload_path = conv_inputpath_to_abspath(dbgym_cfg, default_workload_path(
        dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name
    ))
    # group args. see comment in datagen.py:datagen()
    generic_args = EmbeddingTrainGenericArgs(
        benchmark_name, benchmark_config_path, dataset_path, seed, workload_path
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
