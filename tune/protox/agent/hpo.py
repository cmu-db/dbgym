import numpy as np
import glob
import os
import yaml
import xml.etree.ElementTree as ET
import socket
from pathlib import Path
from tune.protox.agent.wolp.config import _construct_wolp_config, _mutate_wolp_config
from ray import tune


def get_free_port(signal_folder):
    MIN_PORT = 5434
    MAX_PORT = 5500

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = MIN_PORT
    while port <= MAX_PORT:
        try:
            s.bind(('', port))

            drop = False
            for f in glob.glob(f"{signal_folder}/*.signal"):
                if port == int(Path(f).stem):
                    drop = True
                    break

            # Someone else has actually taken hold of this.
            if drop:
                port += 1
                s.close()
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                continue

            with open(f"{signal_folder}/{port}.signal", "w") as f:
                f.write(str(port))
                f.close()

            s.close()
            return port
        except OSError as e:
            port += 1
    raise IOError("No free ports to bind postgres to.")


def _mutate_common_config(logdir, mythril_dir, hpo_config, mythril_args):
    # Copy the benchmark file.
    benchmark_config = mythril_args.benchmark_config
    with open(f"{mythril_dir}/{benchmark_config}") as f:
        benchmark_config = yaml.safe_load(f)
    benchmark = benchmark_config["mythril"]["benchmark"]
    benchmark_config["mythril"]["per_query_knobs"] = hpo_config["mythril_per_query_knobs"]
    benchmark_config["mythril"]["per_query_knob_gen"] = hpo_config["mythril_per_query_knob_gen"]
    benchmark_config["mythril"]["per_query_scan_method"] = hpo_config["mythril_per_query_scan_method"]
    benchmark_config["mythril"]["per_query_select_parallel"] = hpo_config["mythril_per_query_select_parallel"]
    benchmark_config["mythril"]["index_space_aux_type"] = hpo_config["mythril_index_space_aux_type"]
    benchmark_config["mythril"]["index_space_aux_include"] = hpo_config["mythril_index_space_aux_include"]
    benchmark_config["mythril"]["query_spec"] = hpo_config["mythril_query_spec"]

    if benchmark_config["mythril"]["query_spec"]["query_directory"][0] != "/":
        benchmark_config["mythril"]["query_spec"]["query_directory"] = mythril_dir + "/" + benchmark_config["mythril"]["query_spec"]["query_directory"]
    if benchmark_config["mythril"]["query_spec"]["query_order"][0] != "/":
        benchmark_config["mythril"]["query_spec"]["query_order"] = mythril_dir + "/" + benchmark_config["mythril"]["query_spec"]["query_order"]
    with open(f"{benchmark}.yaml", "w") as f:
        yaml.dump(benchmark_config, stream=f, default_flow_style=False)

    # Mutate the config file.
    config = mythril_args.config
    with open(f"{mythril_dir}/{config}", "r") as f:
        config = yaml.safe_load(f)
    pg_path = os.path.expanduser(config["mythril"]["postgres_path"])
    port = get_free_port(pg_path)

    # Update all the paths and metadata needed.
    config["mythril"]["postgres_path"] = pg_path
    config["mythril"]["benchbase_path"] = os.path.expanduser(config["mythril"]["benchbase_path"])

    benchbase_config_path = mythril_args.benchbase_config_path
    conf_etree = ET.parse(benchbase_config_path)
    jdbc = f"jdbc:postgresql://localhost:{port}/benchbase?preferQueryMode=extended"
    conf_etree.getroot().find("url").text = jdbc

    if "oltpr_sf" in mythril_args:
        if conf_etree.getroot().find("scalefactor") is not None:
            conf_etree.getroot().find("scalefactor").text = str(mythril_args.oltp_sf)
        if conf_etree.getroot().find("terminals") is not None:
            conf_etree.getroot().find("terminals").text = str(mythril_args.oltp_num_terminals)
        if conf_etree.getroot().find("works") is not None:
            works = conf_etree.getroot().find("works").find("work")
            if works.find("time") is not None:
                conf_etree.getroot().find("works").find("work").find("time").text = str(mythril_args.oltp_duration)
            if works.find("warmup") is not None:
                conf_etree.getroot().find("works").find("work").find("warmup").text = str(mythril_args.oltp_warmup)
    conf_etree.write("benchmark.xml")
    config["mythril"]["benchbase_config_path"] = str(Path(logdir) / "benchmark.xml")

    config["mythril"]["postgres_data"] = f"pgdata{port}"
    config["mythril"]["postgres_port"] = port
    config["mythril"]["data_snapshot_path"] = "{mythril_dir}/{snapshot}".format(mythril_dir=mythril_dir, snapshot=mythril_args.data_snapshot_path)
    config["mythril"]["tensorboard_path"] = "tboard/"
    config["mythril"]["output_log_path"] = "."
    config["mythril"]["repository_path"] = "repository/"
    config["mythril"]["dump_path"] = "dump.pickle"

    config["mythril"]["default_quantization_factor"] = hpo_config.default_quantization_factor
    config["mythril"]["metric_state"] = hpo_config.metric_state
    config["mythril"]["index_repr"] = hpo_config.index_repr
    config["mythril"]["normalize_state"] = hpo_config.normalize_state
    config["mythril"]["normalize_reward"] = hpo_config.normalize_reward
    config["mythril"]["maximize_state"] = hpo_config.maximize_state
    config["mythril"]["maximize_knobs_only"] = hpo_config.maximize_knobs_only
    config["mythril"]["start_reset"] = hpo_config.start_reset
    config["mythril"]["gamma"] = hpo_config.gamma
    config["mythril"]["grad_clip"] = hpo_config.grad_clip
    config["mythril"]["reward_scaler"] = hpo_config.reward_scaler
    config["mythril"]["workload_timeout_penalty"] = hpo_config.workload_timeout_penalty
    config["mythril"]["workload_eval_mode"] = hpo_config.workload_eval_mode
    config["mythril"]["workload_eval_inverse"] = hpo_config.workload_eval_inverse
    config["mythril"]["workload_eval_reset"] = hpo_config.workload_eval_reset
    config["mythril"]["scale_noise_perturb"] = hpo_config.scale_noise_perturb

    if "index_vae" in hpo_config:
        # Enable index_vae.
        config["mythril"]["index_vae_metadata"]["index_vae"] = hpo_config.index_vae
        config["mythril"]["index_vae_metadata"]["embeddings"] = f"{mythril_dir}/{hpo_config.embeddings}"

    if "lsc_enabled" in hpo_config:
        config["mythril"]["lsc_parameters"]["lsc_enabled"] = hpo_config.lsc_enabled
        config["mythril"]["lsc_parameters"]["lsc_embed"] = hpo_config.lsc_embed
        config["mythril"]["lsc_parameters"]["lsc_shift_initial"] = hpo_config.lsc_shift_initial
        config["mythril"]["lsc_parameters"]["lsc_shift_increment"] = hpo_config.lsc_shift_increment
        config["mythril"]["lsc_parameters"]["lsc_shift_max"] = hpo_config.lsc_shift_max
        config["mythril"]["lsc_parameters"]["lsc_shift_after"] = hpo_config.lsc_shift_after
        config["mythril"]["lsc_parameters"]["lsc_shift_schedule_eps_freq"] = hpo_config.lsc_shift_schedule_eps_freq

    config["mythril"]["system_knobs"] = hpo_config["mythril_system_knobs"]

    with open("config.yaml", "w") as f:
        yaml.dump(config, stream=f, default_flow_style=False)
    return benchmark_config, pg_path, port


def _construct_common_config():
    return {
        # These are command line parameters.
        # Horizon before resetting.
        "horizon": 5,
        "workload_eval_mode": tune.choice(["global_dual", "prev_dual", "all"]),
        "workload_eval_inverse": tune.choice([False, True]),
        "workload_eval_reset": tune.choice([False, True]),

        # Reward.
        "reward": tune.choice(["multiplier", "relative", "cdb_delta"]),

        # These are config.yaml parameters.
        # Default quantization factor to use.
        "default_quantization_factor": 100,

        "metric_state": tune.choice(["metric", "structure"]),
        # Reward scaler
        "reward_scaler": tune.choice([1, 2, 5, 10]),
        "workload_timeout_penalty": tune.choice([1, 2, 4]),

        # Whether to normalize state or not.
        "normalize_state": True,
        # Whether to normalize reward or not.
        "normalize_reward": tune.choice([False, True]),
        # Whether to employ maximize state reset().
        "maximize_state": tune.choice([False, True]),
        "maximize_knobs_only": False,
        "start_reset": tune.sample_from(lambda spc: bool(np.random.choice([False, True])) if spc["config"]["maximize_state"] else False),
        # Discount.
        "gamma": tune.choice([0, 0.9, 0.95, 0.995, 1.0]),
        # Gradient Clipping.
        "grad_clip": tune.choice([1.0, 5.0, 10.0]),
    }


def construct_wolp_config():
    config = _construct_common_config()
    config.update(_construct_wolp_config())
    return config


def mutate_wolp_config(logdir, mythril_dir, hpo_config, mythril_args):
    benchmark, pg_path, port = _mutate_common_config(logdir, mythril_dir, hpo_config, mythril_args)
    _mutate_wolp_config(mythril_dir, hpo_config, mythril_args)
    return benchmark["mythril"]["benchmark"], pg_path, port
