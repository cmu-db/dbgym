from typing import Any

import yaml


def coerce_params(space: dict[str, Any], hpo_config: dict[str, Any]) -> dict[str, Any]:
    if "space_version" not in hpo_config:
        # This is an old version. Coerce the params file.
        new_config = {}
        margs = hpo_config["mythril_args"]

        with open(margs["benchmark_config"]) as f:
            benchmark_config = yaml.safe_load(f)
            benchmark = [k for k in benchmark_config.keys()][0]
            benchmark_config = benchmark_config[benchmark]
            benchmark_config["benchmark"] = benchmark

        # Merge the query specs.
        mqs = hpo_config["mythril_query_spec"]
        benchmark_config["query_spec"].update(mqs)

        defaults = {
            "verbose": True,
            "trace": True,
            "seed": hpo_config["mythril_args"]["seed"],
            "duration": hpo_config["mythril_args"]["duration"],
            "workload_timeout": hpo_config["mythril_args"]["workload_timeout"],
            "query_timeout": hpo_config["mythril_args"]["timeout"],
            "pristine_pgdata_snapshot_path": hpo_config["mythril_args"]["pristine_pgdata_snapshot_path"],
            "output_log_path": "artifacts",
            "pgconn_info": {
                "pgport": 5432,
                "pguser": "admin",
                "pgpass": "",
                "pgdata_path": "/mnt/nvme0n1/wz2/noisepage/pgdata",
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
                "benchbase_config_path": hpo_config["mythril_args"][
                    "benchbase_config_path"
                ],
            },
            "system_knobs": hpo_config["mythril_system_knobs"],
            "lsc": {
                "enabled": hpo_config["lsc_parameters"]["lsc_enabled"],
                "initial": hpo_config["lsc_parameters"]["lsc_shift_initial"],
                "increment": hpo_config["lsc_parameters"]["lsc_shift_increment"],
                "max": hpo_config["lsc_parameters"]["lsc_shift_max"],
                "shift_eps_freq": hpo_config["lsc_parameters"][
                    "lsc_shift_schedule_eps_freq"
                ],
                "shift_after": hpo_config["lsc_parameters"]["lsc_shift_after"],
            },
            "neighbor_parameters": {
                "knob_num_nearest": hpo_config["neighbor_parameters"][
                    "knob_num_nearest"
                ],
                "knob_span": hpo_config["neighbor_parameters"]["knob_span"],
                "index_num_samples": hpo_config["neighbor_parameters"][
                    "index_num_samples"
                ],
                "index_rules": hpo_config["neighbor_parameters"].get(
                    "index_subset", True
                ),
            },
            "embedding_paths": hpo_config["vae_metadata"]["embedding_paths"],
        }

        for s in space.keys():
            if s in defaults:
                new_config[s] = defaults[s]
            elif s in hpo_config:
                new_config[s] = hpo_config[s]
            elif s == "space_version":
                continue
            else:
                assert False, print(f"{s} unable to coerce.")

        return new_config

    return hpo_config
