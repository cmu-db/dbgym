from pathlib import Path


class EmbeddingTrainGenericArgs:
    """Same comment as EmbeddingDatagenGenericArgs"""

    def __init__(
        self,
        benchmark_name: str,
        workload_name: str,
        scale_factor: float,
        benchmark_config_path: Path,
        traindata_path: Path,
        seed: int,
        workload_path: Path,
    ) -> None:
        self.benchmark_name = benchmark_name
        self.workload_name = workload_name
        self.scale_factor = scale_factor
        self.benchmark_config_path = benchmark_config_path
        self.traindata_path = traindata_path
        self.seed = seed
        self.workload_path = workload_path


class EmbeddingTrainAllArgs:
    """Same comment as EmbeddingDatagenGenericArgs"""

    def __init__(
        self,
        hpo_space_path: Path,
        train_max_concurrent: int,
        iterations_per_epoch: int,
        num_samples: int,
        train_size: float,
    ) -> None:
        self.hpo_space_path = hpo_space_path
        self.train_max_concurrent = train_max_concurrent
        self.iterations_per_epoch = iterations_per_epoch
        self.num_samples = num_samples
        self.train_size = train_size


class EmbeddingAnalyzeArgs:
    """Same comment as EmbeddingDatagenGenericArgs"""

    def __init__(
        self,
        start_epoch: int,
        batch_size: int,
        num_batches: int,
        max_segments: int,
        num_points_to_sample: int,
        num_classes_to_keep: int,
    ) -> None:
        self.start_epoch = start_epoch
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.max_segments = max_segments
        self.num_points_to_sample = num_points_to_sample
        self.num_classes_to_keep = num_classes_to_keep


class EmbeddingSelectArgs:
    """Same comment as EmbeddingDatagenGenericArgs"""

    def __init__(
        self,
        recon: float,
        latent_dim: int,
        bias_sep: float,
        idx_limit: int,
        num_curate: int,
        allow_all: bool,
        flatten_idx: int,
    ) -> None:
        self.recon = recon
        self.latent_dim = latent_dim
        self.bias_sep = bias_sep
        self.idx_limit = idx_limit
        self.num_curate = num_curate
        self.allow_all = allow_all
        self.flatten_idx = flatten_idx
