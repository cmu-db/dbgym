import numpy as np
import glob
import os
import yaml
import xml.etree.ElementTree as ET
import socket
from pathlib import Path
from tune.protox.agent.wolp.config import _construct_wolp_config, _mutate_wolp_config
from ray import tune
from misc.utils import open_and_save, conv_inputpath_to_abspath


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


def _mutate_common_config(dbgym_cfg, logdir, protox_dir, hpo_config, protox_args):
    # Copy the benchmark file.
    benchmark_config_path = protox_args.benchmark_config_path
    with open_and_save(dbgym_cfg, benchmark_config_path) as f:
        benchmark_config = yaml.safe_load(f)
    benchmark_name = benchmark_config["protox"]["benchmark_name"]
    benchmark_config["protox"]["per_query_knobs"] = hpo_config["protox_per_query_knobs"]
    benchmark_config["protox"]["per_query_knob_gen"] = hpo_config["protox_per_query_knob_gen"]
    benchmark_config["protox"]["per_query_scan_method"] = hpo_config["protox_per_query_scan_method"]
    benchmark_config["protox"]["per_query_select_parallel"] = hpo_config["protox_per_query_select_parallel"]
    benchmark_config["protox"]["index_space_aux_type"] = hpo_config["protox_index_space_aux_type"]
    benchmark_config["protox"]["index_space_aux_include"] = hpo_config["protox_index_space_aux_include"]
    benchmark_config["protox"]["query_spec"] = hpo_config["protox_query_spec"]

    with open(f"{benchmark_name}.yaml", "w") as f:
        yaml.dump(benchmark_config, stream=f, default_flow_style=False)

    # Mutate the config file.
    protox_config_path = protox_args.protox_config_path
    with open_and_save(dbgym_cfg, protox_config_path, "r") as f:
        protox_config = yaml.safe_load(f)
    pg_path = os.path.expanduser(protox_config["protox"]["postgres_path"])
    port = get_free_port(pg_path)

    # Update all the paths and metadata needed.
    # protox_config will be dumped to a .yaml file, so all Path objects have to be converted to strings
    protox_config["protox"]["postgres_path"] = str(pg_path)
    protox_config["protox"]["benchbase_path"] = str(os.path.expanduser(protox_config["protox"]["benchbase_path"]))

    benchbase_config_path = protox_args.benchbase_config_path
    # usually, we open file with open_and_save. however, this is a special case where we need to call external code (ET.parse) that takes in a file path
    # this means we need to (1) call open_and_save just to save the file and (2) convert the path to an absolute path before sending it to
    #   ET.parse (a conversion normally done by open_and_save)
    open_and_save(dbgym_cfg, benchbase_config_path, "r")
    conf_etree = ET.parse(conv_inputpath_to_abspath(dbgym_cfg, benchbase_config_path))
    jdbc = f"jdbc:postgresql://localhost:{port}/benchbase?preferQueryMode=extended"
    conf_etree.getroot().find("url").text = jdbc

    if "oltpr_sf" in protox_args:
        if conf_etree.getroot().find("scalefactor") is not None:
            conf_etree.getroot().find("scalefactor").text = str(protox_args.oltp_sf)
        if conf_etree.getroot().find("terminals") is not None:
            conf_etree.getroot().find("terminals").text = str(protox_args.oltp_num_terminals)
        if conf_etree.getroot().find("works") is not None:
            works = conf_etree.getroot().find("works").find("work")
            if works.find("time") is not None:
                conf_etree.getroot().find("works").find("work").find("time").text = str(protox_args.oltp_duration)
            if works.find("warmup") is not None:
                conf_etree.getroot().find("works").find("work").find("warmup").text = str(protox_args.oltp_warmup)
    conf_etree.write("benchmark.xml")
    # protox_config will be dumped to a .yaml file, so all Path objects have to be converted to strings
    protox_config["protox"]["benchbase_config_path"] = str(Path(logdir) / "benchmark.xml")

    protox_config["protox"]["postgres_data"] = f"pgdata{port}"
    protox_config["protox"]["postgres_port"] = port
    # protox_config will be dumped to a .yaml file, so all Path objects have to be converted to strings
    protox_config["protox"]["pgdata_snapshot_path"] = str(protox_args.pgdata_snapshot_path)
    protox_config["protox"]["tensorboard_path"] = "tboard/"
    protox_config["protox"]["output_log_path"] = "."
    protox_config["protox"]["repository_path"] = "repository/"
    protox_config["protox"]["dump_path"] = "dump.pickle"

    protox_config["protox"]["default_quantization_factor"] = hpo_config.default_quantization_factor
    protox_config["protox"]["metric_state"] = hpo_config.metric_state
    protox_config["protox"]["index_repr"] = hpo_config.index_repr
    protox_config["protox"]["normalize_state"] = hpo_config.normalize_state
    protox_config["protox"]["normalize_reward"] = hpo_config.normalize_reward
    protox_config["protox"]["maximize_state"] = hpo_config.maximize_state
    protox_config["protox"]["maximize_knobs_only"] = hpo_config.maximize_knobs_only
    protox_config["protox"]["start_reset"] = hpo_config.start_reset
    protox_config["protox"]["gamma"] = hpo_config.gamma
    protox_config["protox"]["grad_clip"] = hpo_config.grad_clip
    protox_config["protox"]["reward_scaler"] = hpo_config.reward_scaler
    protox_config["protox"]["workload_timeout_penalty"] = hpo_config.workload_timeout_penalty
    protox_config["protox"]["workload_eval_mode"] = hpo_config.workload_eval_mode
    protox_config["protox"]["workload_eval_inverse"] = hpo_config.workload_eval_inverse
    protox_config["protox"]["workload_eval_reset"] = hpo_config.workload_eval_reset
    protox_config["protox"]["scale_noise_perturb"] = hpo_config.scale_noise_perturb

    if "index_vae" in hpo_config:
        # Enable index_vae.
        protox_config["protox"]["index_vae_metadata"]["index_vae"] = hpo_config.index_vae

    if "lsc_enabled" in hpo_config:
        protox_config["protox"]["lsc_parameters"]["lsc_enabled"] = hpo_config.lsc_enabled
        protox_config["protox"]["lsc_parameters"]["lsc_embed"] = hpo_config.lsc_embed
        protox_config["protox"]["lsc_parameters"]["lsc_shift_initial"] = hpo_config.lsc_shift_initial
        protox_config["protox"]["lsc_parameters"]["lsc_shift_increment"] = hpo_config.lsc_shift_increment
        protox_config["protox"]["lsc_parameters"]["lsc_shift_max"] = hpo_config.lsc_shift_max
        protox_config["protox"]["lsc_parameters"]["lsc_shift_after"] = hpo_config.lsc_shift_after
        protox_config["protox"]["lsc_parameters"]["lsc_shift_schedule_eps_freq"] = hpo_config.lsc_shift_schedule_eps_freq

    protox_config["protox"]["system_knobs"] = hpo_config["protox_system_knobs"]

    with open("config.yaml", "w") as f:
        yaml.dump(protox_config, stream=f, default_flow_style=False)
    return benchmark_config, pg_path, port


def _construct_common_config(args):
    args.pop("horizon", None)
    args.pop("reward", None)
    args.pop("max_concurrent", None)
    args.pop("num_trials", None)
    args.pop("initial_configs", None)
    args.pop("initial_repeats", None)

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

        # Stash the protox arguments here.
        "protox_args": args,
    }


def construct_wolp_config(args):
    config = _construct_common_config(args)
    config.update(_construct_wolp_config())
    return config


def mutate_wolp_config(dbgym_cfg, logdir, protox_dir, hpo_config, protox_args):
    benchmark_config, pg_path, port = _mutate_common_config(dbgym_cfg, logdir, protox_dir, hpo_config, protox_args)
    _mutate_wolp_config(dbgym_cfg, protox_dir, hpo_config, protox_args)
    return benchmark_config["protox"]["benchmark_name"], pg_path, port
