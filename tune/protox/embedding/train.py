import random
import click
import numpy as np
import logging
import torch

from tune.protox.embedding.train_all import train_all_embeddings
from tune.protox.embedding.analyze import redist_trained_models, analyze_all_embeddings_parts, compute_num_parts, STATS_FNAME, RANGES_FNAME
from tune.protox.embedding.select import select_best_embeddings

from misc.utils import DEFAULT_HPO_SPACE_RELPATH, default_benchmark_config_relpath, default_dataset_path, BENCHMARK_PLACEHOLDER, DATA_PATH_PLACEHOLDER


class EmbeddingGenericArgs:
    '''
    I made Embedding*Args classes to reduce the # of parameters we pass into functions
    I wanted to use classes over dictionaries to enforce which fields are allowed to be present
    I wanted to make multiple classes instead of just one to conceptually separate the different args
    '''
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


class EmbeddingSelectArgs:
    '''Same comment as EmbeddingGenericArgs'''
    def __init__(self, recon, latent_dim, bias_sep, idx_limit, num_curate, allow_all, flatten_idx):
        self.recon = recon
        self.latent_dim = latent_dim
        self.bias_sep = bias_sep
        self.idx_limit = idx_limit
        self.num_curate = num_curate
        self.allow_all = allow_all
        self.flatten_idx = flatten_idx


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

# select args
@click.option("--recon", type=float, default=None, help="The reconstruction error threshold our selected model(s) need to pass.")
@click.option("--latent-dim", type=int, default=None, help="The # of latent dimensions our selected model(s) need to have.")
@click.option("--bias-sep", type=float, default=None, help="The bias separation our selected model(s) need to have.")
@click.option("--idx-limit", type=int, default=15, help="The number of indexes whose errors to compute during _attach().")
@click.option("--num-curate", default=1, help="The number of models to curate") # TODO(wz2): why would we want to curate more than one?
@click.option("--allow-all", is_flag=True, help="Whether to curate within or across parts.")
@click.option("--flatten-idx", default=0, help="TODO(wz2)")

# misc args
@click.option("--seed", default=None, type=int, help="The seed used for all sources of randomness (random, np, torch, etc.). The default is a random value.")

def train(ctx, benchmark, benchmark_config_path, dataset_path, hpo_space_path, train_max_concurrent, iterations_per_epoch, num_samples, train_size, start_epoch, batch_size, num_batches, max_segments, num_points_to_sample, num_classes_to_keep, recon, latent_dim, bias_sep, idx_limit, num_curate, allow_all, flatten_idx, seed):
    '''
    Trains embeddings with num_samples samples of the hyperparameter space. Analyzes the accuracy of all epochs of all hyperparameter
    space samples. Selects the best embedding(s) and packages it as a .pth file in the run_*/ dir.
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
    select_args = EmbeddingSelectArgs(recon, latent_dim, bias_sep, idx_limit, num_curate, allow_all, flatten_idx)

    # run all steps
    train_all_embeddings(ctx, generic_args, train_args)
    num_parts = compute_num_parts(num_samples)
    redist_trained_models(ctx, num_parts)
    analyze_all_embeddings_parts(ctx, num_parts, generic_args, analyze_args)
    select_best_embeddings(ctx, generic_args, select_args)