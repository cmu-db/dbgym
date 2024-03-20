class EmbeddingTrainGenericArgs:
    """Same comment as EmbeddingDatagenGenericArgs"""

    def __init__(
        self, benchmark_name, benchmark_config_path, dataset_path, seed, workload_path
    ):
        self.benchmark_name = benchmark_name
        self.benchmark_config_path = benchmark_config_path
        self.dataset_path = dataset_path
        self.seed = seed
        self.workload_path = workload_path


class EmbeddingTrainAllArgs:
    """Same comment as EmbeddingDatagenGenericArgs"""

    def __init__(
        self,
        hpo_space_path,
        train_max_concurrent,
        iterations_per_epoch,
        num_samples,
        train_size,
    ):
        self.hpo_space_path = hpo_space_path
        self.train_max_concurrent = train_max_concurrent
        self.iterations_per_epoch = iterations_per_epoch
        self.num_samples = num_samples
        self.train_size = train_size


class EmbeddingAnalyzeArgs:
    """Same comment as EmbeddingDatagenGenericArgs"""

    def __init__(
        self,
        start_epoch,
        batch_size,
        num_batches,
        max_segments,
        num_points_to_sample,
        num_classes_to_keep,
    ):
        self.start_epoch = start_epoch
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.max_segments = max_segments
        self.num_points_to_sample = num_points_to_sample
        self.num_classes_to_keep = num_classes_to_keep


class EmbeddingSelectArgs:
    """Same comment as EmbeddingDatagenGenericArgs"""

    def __init__(
        self, recon, latent_dim, bias_sep, idx_limit, num_curate, allow_all, flatten_idx
    ):
        self.recon = recon
        self.latent_dim = latent_dim
        self.bias_sep = bias_sep
        self.idx_limit = idx_limit
        self.num_curate = num_curate
        self.allow_all = allow_all
        self.flatten_idx = flatten_idx
