from typing import Any

import yaml

from misc.utils import DBGymConfig, TuningMode, open_and_save


def coerce_config(
    dbgym_cfg: DBGymConfig, space: dict[str, Any], hpo_params: dict[str, Any]
) -> dict[str, Any]:
    if "space_version" not in hpo_params:
        # This is an old version. Coerce the params file.
        new_config = {}
        margs = hpo_params["mythril_args"]

        with open_and_save(dbgym_cfg, margs["benchmark_config"]) as f:
            benchmark_config = yaml.safe_load(f)
            benchmark = [k for k in benchmark_config.keys()][0]
            benchmark_config = benchmark_config[benchmark]
            benchmark_config["benchmark"] = benchmark

        # Merge the query specs.
        mqs = hpo_params["mythril_query_spec"]
        benchmark_config["query_spec"].update(mqs)

        defaults = {
            "verbose": True,
            "trace": True,
            "seed": hpo_params["mythril_args"]["seed"],
            "tune_duration": {
                str(TuningMode.HPO): hpo_params["mythril_args"]["duration"],
            },
            "workload_timeout": {
                str(TuningMode.HPO): hpo_params["mythril_args"]["workload_timeout"],
            },
            "query_timeout": hpo_params["mythril_args"]["timeout"],
            "pgconn_info": {
                "pgport": 5432,
                "pguser": "admin",
                "pgpass": "",
                "pristine_dbdata_snapshot_path": "/mnt/nvme0n1/wz2/noisepage/pgdata",
                "dbdata_parent_dpath": "/mnt/nvme0n1/wz2/noisepage/",
                "pgbin_path": "/mnt/nvme0n1/wz2/noisepage/",
            },
            "benchmark_config": benchmark_config,
            "benchbase_config": {
                "oltp_config": {
                    "oltp_num_terminals": margs.get("oltp_num_terminals", 0),
                    "oltp_duration": margs.get("oltp_duration", 0),
                    "oltp_sf": margs.get("oltp_sf", 0),
                    "oltp_warmup": margs.get("oltp_warmup", 0),
                },
                "benchbase_path": "/home/wz2/noisepage-pilot/artifacts/benchbase/",
                "benchbase_config_path": hpo_params["mythril_args"][
                    "benchbase_config_path"
                ],
            },
            "system_knobs": hpo_params["mythril_system_knobs"],
            "lsc": {
                "enabled": hpo_params["lsc_parameters"]["lsc_enabled"],
                "initial": hpo_params["lsc_parameters"]["lsc_shift_initial"],
                "increment": hpo_params["lsc_parameters"]["lsc_shift_increment"],
                "max": hpo_params["lsc_parameters"]["lsc_shift_max"],
                "shift_eps_freq": hpo_params["lsc_parameters"][
                    "lsc_shift_schedule_eps_freq"
                ],
                "shift_after": hpo_params["lsc_parameters"]["lsc_shift_after"],
            },
            "neighbor_parameters": {
                "knob_num_nearest": hpo_params["neighbor_parameters"][
                    "knob_num_nearest"
                ],
                "knob_span": hpo_params["neighbor_parameters"]["knob_span"],
                "index_num_samples": hpo_params["neighbor_parameters"][
                    "index_num_samples"
                ],
                "index_rules": hpo_params["neighbor_parameters"].get(
                    "index_subset", True
                ),
            },
            "embedder_path": hpo_params["vae_metadata"]["embedder_path"],
        }

        for s in space.keys():
            if s in defaults:
                new_config[s] = defaults[s]
            elif s in hpo_params:
                new_config[s] = hpo_params[s]
            elif s == "space_version":
                continue
            else:
                assert False, print(f"{s} unable to coerce.")

        return new_config

    return hpo_params
