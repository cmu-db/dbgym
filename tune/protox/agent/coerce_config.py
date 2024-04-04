from typing import Any
import yaml

from misc.utils import DBGymConfig, open_and_save


def coerce_config(dbgym_cfg: DBGymConfig, space: dict[str, Any], hpoed_params: dict[str, Any]) -> dict[str, Any]:
    if "space_version" not in hpoed_params:
        # This is an old version. Coerce the params file.
        new_config = {}
        margs = hpoed_params["mythril_args"]

        with open_and_save(dbgym_cfg, margs["benchmark_config"]) as f:
            benchmark_config = yaml.safe_load(f)
            benchmark = [k for k in benchmark_config.keys()][0]
            benchmark_config = benchmark_config[benchmark]
            benchmark_config["benchmark"] = benchmark

        # Merge the query specs.
        mqs = hpoed_params["mythril_query_spec"]
        benchmark_config["query_spec"].update(mqs)

        defaults = {
            "verbose": True,
            "trace": True,
            "seed": hpoed_params["mythril_args"]["seed"],
            "duration": hpoed_params["mythril_args"]["duration"],
            "workload_timeout": hpoed_params["mythril_args"]["workload_timeout"],
            "query_timeout": hpoed_params["mythril_args"]["timeout"],
            "output_log_path": "artifacts",
            "pgconn_info": {
                "pgport": 5432,
                "pguser": "admin",
                "pgpass": "",
                "pristine_pgdata_snapshot_path": "/mnt/nvme0n1/wz2/noisepage/pgdata",
                "pgdata_parent_dpath": "/mnt/nvme0n1/wz2/noisepage/",
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
                "benchbase_config_path": hpoed_params["mythril_args"][
                    "benchbase_config_path"
                ],
            },
            "system_knobs": hpoed_params["mythril_system_knobs"],
            "lsc": {
                "enabled": hpoed_params["lsc_parameters"]["lsc_enabled"],
                "initial": hpoed_params["lsc_parameters"]["lsc_shift_initial"],
                "increment": hpoed_params["lsc_parameters"]["lsc_shift_increment"],
                "max": hpoed_params["lsc_parameters"]["lsc_shift_max"],
                "shift_eps_freq": hpoed_params["lsc_parameters"][
                    "lsc_shift_schedule_eps_freq"
                ],
                "shift_after": hpoed_params["lsc_parameters"]["lsc_shift_after"],
            },
            "neighbor_parameters": {
                "knob_num_nearest": hpoed_params["neighbor_parameters"][
                    "knob_num_nearest"
                ],
                "knob_span": hpoed_params["neighbor_parameters"]["knob_span"],
                "index_num_samples": hpoed_params["neighbor_parameters"][
                    "index_num_samples"
                ],
                "index_rules": hpoed_params["neighbor_parameters"].get(
                    "index_subset", True
                ),
            },
            "embedder_path": hpoed_params["vae_metadata"]["embedder_path"],
        }

        for s in space.keys():
            if s in defaults:
                new_config[s] = defaults[s]
            elif s in hpoed_params:
                new_config[s] = hpoed_params[s]
            elif s == "space_version":
                continue
            else:
                assert False, print(f"{s} unable to coerce.")

        return new_config

    return hpoed_params
