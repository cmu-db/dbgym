import gc
import itertools
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
import yaml

from misc.utils import open_and_save
from tune.protox.embedding.loss import CostLoss, get_bias_fn
from tune.protox.embedding.trainer import StratifiedRandomSampler
from tune.protox.embedding.train_all import fetch_index_parameters, load_input_data, create_vae_model
from tune.protox.embedding.vae import VAELoss, gen_vae_collate
from tune.protox.env.space.primitive_space.index_space import IndexSpace

STATS_FNAME = "stats.txt"
RANGES_FNAME = "ranges.txt"


def compute_num_parts(num_samples):
    # TODO(phw2): in the future, implement running different parts in parallel, set OMP_NUM_THREADS accordingly, and investigate the effect of having more parts
    # TODO(phw2): if having more parts is effective, figure out a good way to specify num_parts (can it be determined automatically or should it be a CLI arg?)
    # TODO(phw2): does anything bad happen if num_parts doesn't evenly divide num_samples?
    return 1


def redist_trained_models(dbgym_cfg, num_parts):
    """
    Redistribute all embeddings_*/ folders inside the run_*/ folder into num_parts subfolders
    """
    inputs = [
        f for f in dbgym_cfg.dbgym_this_run_path.glob("embeddings*") if os.path.isdir(f)
    ]

    for part_i in range(num_parts):
        Path(_get_part_i_dpath(dbgym_cfg, part_i)).mkdir(parents=True, exist_ok=True)

    for model_i, emb in enumerate(inputs):
        part_i = model_i % num_parts
        shutil.move(emb, _get_part_i_dpath(dbgym_cfg, part_i))


def analyze_all_embeddings_parts(dbgym_cfg, num_parts, generic_args, analyze_args):
    """
    Analyze all part*/ dirs _in parallel_
    """
    start_time = time.time()
    for part_i in range(num_parts):
        _analyze_embeddings_part(dbgym_cfg, part_i, generic_args, analyze_args)
    duration = time.time() - start_time
    with open(
        os.path.join(dbgym_cfg.dbgym_this_run_path, "analyze_all_time.txt"), "w"
    ) as f:
        f.write(f"{duration}")


def _analyze_embeddings_part(dbgym_cfg, part_i, generic_args, analyze_args):
    """
    Analyze (meaning create both stats.txt and ranges.txt) all the embedding models in the part[part_i]/ dir
    """
    part_dpath = _get_part_i_dpath(dbgym_cfg, part_i)

    start_time = time.time()
    _create_stats_for_part(dbgym_cfg, part_dpath, generic_args, analyze_args)
    duration = time.time() - start_time
    with open(os.path.join(part_dpath, "stats_time.txt"), "w") as f:
        f.write(f"{duration}")

    start_time = time.time()
    _create_ranges_for_part(dbgym_cfg, part_dpath, generic_args, analyze_args)
    duration = time.time() - start_time
    with open(os.path.join(part_dpath, "ranges_time.txt"), "w") as f:
        f.write(f"{duration}")


def _create_stats_for_part(dbgym_cfg, part_dpath, generic_args, analyze_args):
    """
    Creates a stats.txt file inside each embeddings_*/models/epoch*/ dir inside this part*/ dir
    TODO(wz2): what does stats.txt contain?
    """
    # Unlike for training, we're safe to use all threads for creating stats
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

    # Load the benchmark configuration.
    with open_and_save(dbgym_cfg, generic_args.benchmark_config_path, "r") as f:
        data = yaml.safe_load(f)
        max_attrs, max_cat_features, _, _ = fetch_index_parameters(
            dbgym_cfg, generic_args.benchmark_name, data, generic_args.workload_path
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    models = itertools.chain(*[Path(part_dpath).rglob("config")])
    models = [m for m in models]
    for model_config in tqdm.tqdm(models):
        if ((Path(model_config).parent) / "FAILED").exists():
            print("Detected failure in: ", model_config)
            continue

        with open_and_save(dbgym_cfg, model_config, "r") as f:
            config = json.load(f)

        # Create them here since these are constant for a given "model" configuration.
        dataset, idx_class, num_classes = None, None, None
        class_mapping = None
        metric_loss_fn, vae_loss = None, None
        vae = create_vae_model(config, max_attrs, max_cat_features)
        require_cost = config["metric_loss_md"].get("require_cost", False)

        submodules = [f for f in (Path(model_config).parent / "models").glob("*")]
        submodules = sorted(submodules, key=lambda x: int(str(x).split("epoch")[-1]))
        # This is done for semantic sense since the "first" is actually at no epoch.
        modules = [submodules[r] for r in range(-1, len(submodules)) if r >= 0]
        if modules[0] != submodules[0]:
            modules = [submodules[0]] + modules

        if modules[-1] != submodules[-1]:
            modules.append(submodules[-1])

        modules = [
            m
            for m in modules
            if int(str(m).split("epoch")[-1]) >= analyze_args.start_epoch
        ]

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
                dataset, _, idx_class, _, num_classes = load_input_data(
                    dbgym_cfg,
                    generic_args.dataset_path,
                    1.0,
                    max_attrs,
                    require_cost,
                    seed=0,
                )

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
            if analyze_all_batches or analyze_args.num_batches > 0:
                accumulated_stats.update(
                    {
                        "recon_accum": [],
                        "metric_accum": [],
                    }
                )

                # Setup the dataloader.
                if analyze_all_batches:
                    dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=analyze_args.batch_size,
                        collate_fn=collate_fn,
                    )
                    total = len(dataloader)
                else:
                    sampler = StratifiedRandomSampler(
                        idx_class,
                        max_class=num_classes,
                        batch_size=analyze_args.batch_size,
                        allow_repeats=False,
                    )
                    dataloader = torch.utils.data.DataLoader(
                        dataset,
                        sampler=sampler,
                        batch_size=analyze_args.batch_size,
                        collate_fn=collate_fn,
                    )
                    total = min(len(sampler), analyze_args.num_batches)
                error = False
                with torch.no_grad():
                    with tqdm.tqdm(total=total, leave=False) as pbar:
                        for x, y in dataloader:
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
                                is_eval=True,
                            )

                            assert vae_loss.loss_fn is not None
                            for class_idx in class_mapping:
                                y_mask = classes == class_idx
                                x_extract = x[y_mask.bool()]
                                if x_extract.shape[0] > 0:
                                    decoded_extract = decoded[y_mask.bool()]
                                    loss = vae_loss.loss_fn(
                                        decoded_extract, x_extract, y[y_mask.bool()]
                                    )
                                    accumulated_stats[f"recon_{class_idx}"].append(
                                        loss.mean().item()
                                    )

                            input_y = y
                            if y.shape[1] == 1:
                                input_y = y.flatten()

                            metric_loss = metric_loss_fn(z, input_y, None).item()
                            accumulated_stats["recon_accum"].append(
                                loss_dict["recon_loss"]["losses"].item()
                            )
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
                        stat_key: (
                            stats
                            if isinstance(stats, np.ScalarType)
                            else (np.mean(stats) if len(stats) > 0 else 0)
                        )
                        for stat_key, stats in accumulated_stats.items()
                    }
                    stats["error"] = error.item()
                    f.write(json.dumps(stats, indent=4))

                del dataloader
                gc.collect()
                gc.collect()


def _create_ranges_for_part(dbgym_cfg, part_dpath, generic_args, analyze_args):
    """
    Create the ranges.txt for all models in part_dpath
    TODO(wz2): what does ranges.txt contain?
    """
    # Unlike for training, we're safe to use all threads for creating ranges
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    paths = sorted(
        [
            f
            for f in Path(part_dpath).rglob("embedder_*.pth")
            if "optimizer" not in str(f)
        ]
    )
    for embedder_fpath in tqdm.tqdm(paths):
        _create_ranges_for_embedder(
            dbgym_cfg, embedder_fpath, generic_args, analyze_args
        )


def _create_ranges_for_embedder(dbgym_cfg, embedder_fpath, generic_args, analyze_args):
    """
    Create the ranges.txt file corresponding to a specific part*/embeddings_*/models/epoch*/embedder_*.pth file
    """
    # Return right away if the epoch isn't high enough
    epoch_i = int(str(embedder_fpath).split("embedder_")[-1].split(".pth")[0])
    if epoch_i < analyze_args.start_epoch:
        return

    # Load the benchmark configuration.
    with open_and_save(dbgym_cfg, generic_args.benchmark_config_path, "r") as f:
        data = yaml.safe_load(f)
        tables = data["protox"]["tables"]
        max_attrs, max_cat_features, att_usage, _ = fetch_index_parameters(
            dbgym_cfg, generic_args.benchmark_name, data, generic_args.workload_path
        )

    # don't use open_and_save() because we generated embeddings_config_fpath in this run
    embeddings_dpath = embedder_fpath.parent.parent.parent  # part*/embeddings_*/
    embeddings_config_fpath = os.path.join(
        embeddings_dpath, "config"
    )  # part*/embeddings_*/config
    with open(embeddings_config_fpath, "r") as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = create_vae_model(config, max_attrs, max_cat_features)
    # Load the specific epoch model.
    vae.load_state_dict(torch.load(embedder_fpath, map_location=device))
    vae.to(device=device).eval()

    idxs = IndexSpace(
        agent_type="wolp",
        tables=tables,
        max_num_columns=0,
        # index_repr=IndexRepr.ONE_HOT_DETERMINISTIC.name, TODO(phw2)
        seed=np.random.randint(1, 1e10),
        latent_dim=config["latent_dim"],
        index_vae_model=vae,
        index_output_scale=1.0,
        attributes_overwrite=att_usage,
    )
    idxs.rel_metadata = att_usage
    idxs._build_mapping(att_usage)

    def decode_to_classes(rand_points):
        with torch.no_grad():
            rand_decoded = idxs._decode(act=rand_points)
            classes = {}
            for r in range(rand_points.shape[0]):
                act = idxs.index_repr_policy.sample_action(
                    idxs.np_random, rand_decoded[r], att_usage, False, True
                )
                idx_class = idxs.get_index_class(act)
                if idx_class not in classes:
                    classes[idx_class] = 0
                classes[idx_class] += 1
        return sorted(
            [(k, v) for k, v in classes.items()], key=lambda x: x[1], reverse=True
        )

    output_scale = config["metric_loss_md"]["output_scale"]
    bias_separation = config["metric_loss_md"]["bias_separation"]
    num_segments = min(analyze_args.max_segments, math.ceil(1.0 / bias_separation))

    base = 0
    epoch_dpath = os.path.join(
        embeddings_dpath, "models", f"epoch{epoch_i}"
    )  # part*/embeddings_*/models/epoch*/
    ranges_fpath = os.path.join(epoch_dpath, RANGES_FNAME)
    with open(ranges_fpath, "w") as f:
        for _ in tqdm.tqdm(range(num_segments), total=num_segments, leave=False):
            classes = decode_to_classes(
                torch.rand(analyze_args.num_points_to_sample, config["latent_dim"])
                * output_scale
                + base
            )
            if analyze_args.num_classes_to_keep != 0:
                classes = classes[: analyze_args.num_classes_to_keep]

            f.write(f"Generating range {base} - {base + output_scale}\n")
            f.write(
                "\n".join(
                    [
                        f"{k}: {v / analyze_args.num_points_to_sample}"
                        for (k, v) in classes
                    ]
                )
            )
            f.write("\n")
            base += output_scale


def _get_part_i_dpath(dbgym_cfg, part_i) -> str:
    return os.path.join(dbgym_cfg.dbgym_this_run_path, f"part{part_i}")
