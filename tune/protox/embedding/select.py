import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import tqdm
from pandas import DataFrame

from misc.utils import DBGymConfig, default_embedder_dname, link_result
from tune.protox.embedding.analyze import RANGES_FNAME, STATS_FNAME
from tune.protox.embedding.train_args import (
    EmbeddingSelectArgs,
    EmbeddingTrainGenericArgs,
)


def select_best_embeddings(
    dbgym_cfg: DBGymConfig,
    generic_args: EmbeddingTrainGenericArgs,
    select_args: EmbeddingSelectArgs,
) -> None:
    data = _load_data(dbgym_cfg, select_args)

    if generic_args.traindata_path is not None and generic_args.traindata_path.exists():
        raw_data = pd.read_parquet(generic_args.traindata_path)
        data = _attach(data, raw_data, select_args.idx_limit)

    curated_dpath = dbgym_cfg.cur_task_runs_data_path("curated", mkdir=True)
    curated_results_fpath = (
        dbgym_cfg.cur_task_runs_data_path(mkdir=True) / "curated_results.csv"
    )
    data.to_csv(curated_results_fpath, index=False)

    if "idx_class_total_error" in data:
        data["elbo"] = data.elbo + data.idx_class_total_error

    if select_args.allow_all:
        df = data.sort_values(by=["elbo"]).iloc[: select_args.num_curate]
    else:
        df = (
            data.sort_values(by=["elbo"])
            .groupby(by=["root"])
            .head(1)
            .iloc[: select_args.num_curate]
        )

    if select_args.flatten_idx == -1:
        for tup in df.itertuples():
            assert type(tup.path) is str
            assert type(tup.root) is str
            shutil.copytree(
                tup.path,
                curated_dpath / tup.path,
                dirs_exist_ok=True,
            )
            shutil.copy(
                Path(tup.root) / "config",
                curated_dpath / tup.root / "config",
            )
    else:
        idx = select_args.flatten_idx
        info_txt = open(curated_dpath / "info.txt", "w")

        for loop_i, tup in enumerate(df.itertuples()):
            assert type(tup.path) is str
            assert type(tup.root) is str
            epoch = int(str(tup.path).split("epoch")[-1])
            model_dpath = curated_dpath / f"model{idx}"
            shutil.copytree(tup.path, model_dpath)
            shutil.copy(
                Path(tup.root) / "config",
                model_dpath / "config",
            )
            shutil.move(
                model_dpath / f"embedder_{epoch}.pth",
                model_dpath / "embedder.pth",
            )

            if loop_i == 0:
                link_result(
                    dbgym_cfg,
                    model_dpath,
                    custom_result_name=default_embedder_dname(
                        generic_args.benchmark_name, generic_args.workload_name
                    )
                    + ".link",
                )

            info_txt.write(f"model{idx}/embedder.pth\n")
            idx += 1

        info_txt.close()


def _load_data(dbgym_cfg: DBGymConfig, select_args: EmbeddingSelectArgs) -> DataFrame:
    stat_infos = []
    stats = [s for s in dbgym_cfg.dbgym_this_run_path.rglob(STATS_FNAME)]
    print(f"stats={stats}")
    for stat in stats:
        if "curated" in str(stat):
            continue

        info = {}
        # don't use open_and_save() because we generated stat in this run
        with open(stat, "r") as f:
            stat_dict = json.load(f)
            info["recon"] = stat_dict["recon_accum"]
            info["metric"] = stat_dict["metric_accum"]
            info["elbo"] = info["recon"]
            info["elbo_metric"] = info["recon"] + info["metric"]
            info["all_loss"] = info["recon"] + info["metric"]

            if select_args.recon is not None and select_args.recon < info["recon"]:
                # Did not pass reconstruction threshold.
                continue

            info["path"] = str(stat.parent)
            info["root"] = str(stat.parent.parent.parent)

        # don't use open_and_save() because we generated config in this run
        with open(stat.parent.parent.parent / "config", "r") as f:
            config = json.load(f)

            def recurse_set(source: dict[Any, Any], target: dict[Any, Any]) -> None:
                for k, v in source.items():
                    if isinstance(v, dict):
                        recurse_set(v, target)
                    else:
                        target[k] = v

            recurse_set(config, info)
            if select_args.latent_dim is not None:
                if info["latent_dim"] != select_args.latent_dim:
                    continue

            output_scale = config["metric_loss_md"]["output_scale"]
            bias_sep = config["metric_loss_md"]["bias_separation"]

            if select_args.bias_sep is not None:
                if select_args.bias_sep != bias_sep:
                    continue

            info["ranges_file"] = str(Path(stat).parent / RANGES_FNAME)

        stat_infos.append(info)

    data = DataFrame(stat_infos)
    data = data.loc[:, ~(data == data.iloc[0]).all()]

    if "output_scale" not in data:
        data["output_scale"] = output_scale

    if "bias_separation" not in data:
        data["bias_separation"] = bias_sep

    return data


def _attach(data: DataFrame, raw_data: DataFrame, num_limit: int = 0) -> DataFrame:
    # As the group index goes up, the perf should go up (i.e., bounds should tighten)
    filtered_data: dict[tuple[float, float], DataFrame] = {}
    new_data = []
    for tup in tqdm.tqdm(data.itertuples(), total=data.shape[0]):
        tup_dict = {k: getattr(tup, k) for k in data.columns}
        if raw_data is not None and Path(tup_dict["ranges_file"]).exists():

            def compute_dist_score(
                current_dists: dict[str, float], base: float, upper: float
            ) -> float:
                nonlocal filtered_data
                key = (base, upper)
                if key not in filtered_data:
                    data_range = raw_data[
                        (raw_data.quant_mult_cost_improvement >= base)
                        & (raw_data.quant_mult_cost_improvement < upper)
                    ]
                    filtered_data[key] = data_range
                    if data_range.shape[0] == 0:
                        return 0
                else:
                    data_range = filtered_data[key]

                error = 0
                if "real_idx_class" in data_range:
                    data_dists = (
                        data_range.real_idx_class.value_counts() / data_range.shape[0]
                    )
                else:
                    data_dists = (
                        data_range.idx_class.value_counts() / data_range.shape[0]
                    )

                for key, dist in zip(data_dists.index, data_dists):
                    if str(key) not in current_dists:
                        error += dist
                    else:
                        error += abs(current_dists[str(key)] - dist)
                return error

            # don't use open_and_save() because we generated ranges in this run
            with open(tup_dict["ranges_file"], "r") as f:
                errors: list[float] = []
                drange: tuple[Optional[float], Optional[float]] = (None, None)
                current_dists: dict[str, float] = {}

                for line in f:
                    if "Generating range" in line:
                        if len(current_dists) > 0:
                            assert drange[0] is not None
                            assert drange[1] is not None
                            errors.append(
                                compute_dist_score(current_dists, drange[0], drange[1])
                            )
                            if num_limit > 0 and len(errors) >= num_limit:
                                current_dists = {}
                                break

                        if drange[0] is None:
                            drange = (1.0 - tup_dict["bias_separation"], 1.01)
                        else:
                            drange = (
                                drange[0] - tup_dict["bias_separation"],
                                drange[0],
                            )
                        current_dists = {}

                    else:
                        ci = line.split(": ")[0]
                        dist = float(line.strip().split(": ")[-1])
                        current_dists[ci] = dist

                if len(current_dists) > 0:
                    # Put the error in.
                    errors.append(
                        compute_dist_score(
                            current_dists, 0.0, tup_dict["bias_separation"]
                        )
                    )

                tup_dict["idx_class_errors"] = ",".join(
                    [str(np.round(e, 2)) for e in errors]
                )
                for i, e in enumerate(errors):
                    tup_dict[f"idx_class_error{i}"] = np.round(e, 2)

                if len(errors) > 0:
                    tup_dict["idx_class_mean_error"] = np.mean(errors)
                    tup_dict["idx_class_total_error"] = np.sum(errors)
                    tup_dict["idx_class_min_error"] = np.min(errors)
                    tup_dict["idx_class_max_error"] = np.max(errors)
        new_data.append(tup_dict)
    return DataFrame(new_data)
