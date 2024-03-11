import json
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
import time
import click

import ray
from ray.tune import TuneConfig
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune import Trainable, SyncConfig
from ray.air import RunConfig, FailureConfig

from tune.protox.agent.hpo import construct_wolp_config
from misc.utils import open_and_save, DEFAULT_SYSTEM_KNOB_CONFIG_RELPATH, default_benchmark_config_relpath, BENCHMARK_PLACEHOLDER, default_hpoed_agent_params_path, SYMLINKS_PATH_PLACEHOLDER


@click.command()
@click.pass_context
@click.argument("benchmark")
@click.argument("workload-name")
@click.option(
    "--benchmark-config-path",
    default=None,
    type=Path,
    help=f"The path to the .yaml config file for the benchmark. The default is {default_benchmark_config_relpath(BENCHMARK_PLACEHOLDER)}.",
)
@click.option("--system-knob-config-path", default=DEFAULT_SYSTEM_KNOB_CONFIG_RELPATH, type=Path, help=f"The path to the file configuring the ranges and quantization of system knobs.")
@click.option("--hpoed-agent-params-path", default=None, type=Path, help=f"The path to the agent params found by the HPO process. The default is {default_hpoed_agent_params_path(SYMLINKS_PATH_PLACEHOLDER)}.")
@click.option("--max-hpo-concurrent", default=1, type=int, help=f"The max # of concurrent agent models to train during hyperparameter optimization. This is usually set lower than `nproc` to reduce memory pressure.")
@click.option(
    "--num-samples",
    default=40,
    help=f"The # of times to specific hyperparameter configs to sample from the hyperparameter search space and train agent models with.",
)
def train(ctx, benchmark, workload_name, benchmark_config_path, system_knob_config_path, hpoed_agent_params_path, max_hpo_concurrent, num_samples):
    # set args to defaults programmatically (do this before doing anything else in the function)
    cfg = ctx.obj
    # TODO(phw2): figure out whether different scale factors use the same config
    # TODO(phw2): figure out what parts of the config should be taken out (like stuff about tables)
    if benchmark_config_path == None:
        benchmark_config_path = default_benchmark_config_relpath(benchmark)
    if hpoed_agent_params_path == None:
        hpoed_agent_params_path = default_hpoed_agent_params_path(cfg.dbgym_symlinks_path)

    # Get the system knobs.
    with open_and_save(cfg, system_knob_config_path, "r") as f:
        system_knobs = yaml.safe_load(f)["system_knobs"]

    # Per query knobs.
    with open_and_save(cfg, benchmark_config_path, "r") as f:
        bb_config = yaml.safe_load(f)["protox"]
        per_query_scan_method = bb_config["per_query_scan_method"]
        per_query_select_parallel = bb_config["per_query_select_parallel"]
        index_space_aux_type = bb_config["index_space_aux_type"]
        index_space_aux_include = bb_config["index_space_aux_include"]
        per_query_knobs = bb_config["per_query_knobs"]
        per_query_knob_gen = bb_config["per_query_knob_gen"]
        query_spec = bb_config["query_spec"]
        is_oltp = bb_config["oltp_workload"]

    # Connect to cluster or die.
    ray.init(address="localhost:6379", log_to_driver=False)

    # Config.
    config = construct_wolp_config()

    # Pass the knobs through.
    config["protox_system_knobs"] = system_knobs
    config["protox_per_query_knobs"] = per_query_knobs
    config["protox_per_query_scan_method"] = per_query_scan_method
    config["protox_per_query_select_parallel"] = per_query_select_parallel
    config["protox_index_space_aux_type"] = index_space_aux_type
    config["protox_index_space_aux_include"] = index_space_aux_include
    config["protox_per_query_knob_gen"] = per_query_knob_gen
    config["protox_query_spec"] = query_spec

    # Scheduler.
    scheduler = FIFOScheduler()

    hpoed_agent_params = None
    if hpoed_agent_params_path is not None and hpoed_agent_params_path.exists():
        tmp_hpoed_agent_params = []
        with open_and_save(hpoed_agent_params_path, "r") as f:
            hpoed_agent_params = json.load(f)

            for config in hpoed_agent_params:
                if "protox_per_query_knobs" not in config:
                    config["protox_per_query_knobs"] = per_query_knobs
                if "protox_per_query_scan_method" not in config:
                    config["protox_per_query_scan_method"] = per_query_scan_method
                if "protox_per_query_select_parallel" not in config:
                    config["protox_per_query_select_parallel"] = per_query_select_parallel
                if "protox_index_space_aux_type" not in config:
                    config["protox_index_space_aux_type"] = index_space_aux_type
                if "protox_index_space_aux_include" not in config:
                    config["protox_index_space_aux_include"] = index_space_aux_include
                if "protox_per_query_knob_gen" not in config:
                    config["protox_per_query_knob_gen"] = per_query_knob_gen
                if "protox_system_knobs" not in config:
                    config["protox_system_knobs"] = system_knobs

                assert "protox_args" in config
                config["protox_args"]["early_kill"] = early_kill

                for _ in range(initial_repeats):
                    tmp_hpoed_agent_params.append(config)
        hpoed_agent_params = tmp_hpoed_agent_params

    # Search.
    # if hpoed_agent_params == None, hyperparameter optimization will be performend
    # if hpoed_agent_params != None, we will just run a single tuning job with the params hpoed_agent_params
    search = BasicVariantGenerator(points_to_evaluate=hpoed_agent_params, max_concurrent=max_hpo_concurrent)

    # for OLTP, we're trying to *maximize* txn / sec. for OLAP, we're trying to *minimize* total runtime
    mode = "max" if is_oltp else "min"
    tune_config = TuneConfig(
        scheduler=scheduler,
        search_alg=search,
        num_samples=num_samples,
        max_concurrent_trials=max_hpo_concurrent,
        chdir_to_trial_dir=True,
        metric=METRIC,
        mode=mode,
    )

    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_config = RunConfig(
        name=f"ProtoxHPO_{dtime}",
        failure_config=FailureConfig(max_failures=0, fail_fast=True),
        sync_config=SyncConfig(upload_dir=None, syncer=None),
        verbose=2,
        log_to_file=True,
    )

    tuner = ray.tune.Tuner(
        TuneOpt,
        tune_config=tune_config,
        run_config=run_config,
        param_space=config,
    )

    results = tuner.fit()
    if results.num_errors > 0:
        print("Encountered exceptions!")
        for i in range(len(results)):
            if results[i].error:
                print(f"Trial {results[i]} FAILED")
        assert False
    print("Best hyperparameters found were: ", results.get_best_result(metric=METRIC, mode="max").config)


METRIC = "Best Metric"


class DotDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TuneOpt(Trainable):
    def f_unpack_dict(self, dct):
        """
        Unpacks all sub-dictionaries in given dictionary recursively.
        There should be no duplicated keys across all nested
        subdictionaries, or some instances will be lost without warning

        Source: https://www.kaggle.com/fanvacoolt/tutorial-on-hyperopt

        Parameters:
        ----------------
        dct : dictionary to unpack

        Returns:
        ----------------
        : unpacked dictionary
        """
        res = {}
        for (k, v) in dct.items():
            if "protox_" in k:
                res[k] = v
            elif isinstance(v, dict):
                res = {**res, **self.f_unpack_dict(v)}
            else:
                res[k] = v
        return res


    def setup(self, hpo_config):
        print("HPO Configuration: ", hpo_config)
        assert "protox_args" in hpo_config
        protox_args = hpo_config["protox_args"]
        protox_dir = os.path.expanduser(protox_args["protox_dir"])
        sys.path.append(protox_dir)

        from tune import TuneTrial, TimeoutChecker
        from tune.protox.tune.hpo import mutate_wolp_config
        hpo_config = DotDict(self.f_unpack_dict(hpo_config))
        protox_args = DotDict(self.f_unpack_dict(protox_args))

        # Compute the limit.
        self.early_kill = protox_args["early_kill"]
        self.stabilize_kill = 0
        if "stabilize_kill" in protox_args:
            self.stabilize_kill = protox_args["stabilize_kill"]

        self.last_best_time = None
        self.last_best_metric = None

        self.duration = protox_args["duration"] * 3600
        self.workload_timeout = protox_args["workload_timeout"]
        self.timeout = TimeoutChecker(protox_args["duration"])
        if protox_args.agent == "wolp":
            benchmark, pg_path, port = mutate_wolp_config(self.logdir, protox_dir, hpo_config, protox_args)
        else:
            assert False, f"Unspecified agent {protox_args.agent}"

        self.pg_path = pg_path
        self.port = port

        # We will now overwrite the config files.
        protox_args["config"] = str(Path(self.logdir) / "config.yaml")
        protox_args["model_config"] = str(Path(self.logdir) / "model_params.yaml")
        protox_args["benchmark_config"] = str(Path(self.logdir) / f"{benchmark}.yaml")
        protox_args["reward"] = hpo_config.reward
        protox_args["horizon"] = hpo_config.horizon
        self.trial = TuneTrial()
        self.trial.setup(protox_args, self.timeout)
        self.start_time = time.time()

    def step(self):
        self.timeout.resume()
        data = self.trial.step()

        # Decrement remaining time.
        self.timeout.pause()
        if self.timeout():
            self.cleanup()
            data[ray.tune.result.DONE] = True

        if self.early_kill:
            if (time.time() - self.start_time) >= 10800:
                if "Best Metric" in data and data["Best Metric"] >= 190:
                    self.cleanup()
                    data[ray.tune.result.DONE] = True
            elif (time.time() - self.start_time) >= 7200:
                if "Best Metric" in data and data["Best Metric"] >= 250:
                    self.cleanup()
                    data[ray.tune.result.DONE] = True

        if self.stabilize_kill > 0 and "Best Metric" in data:
            if self.last_best_metric is None or data["Best Metric"] < self.last_best_metric:
                self.last_best_metric = data["Best Metric"]
                self.last_best_time = time.time()

            if self.last_best_time is not None and (time.time() - self.last_best_time) > self.stabilize_kill * 3600:
                self.trial.logger.info("Killing due to run stabilizing.")
                self.cleanup()
                data[ray.tune.result.DONE] = True

        return data

    def cleanup(self):
        self.trial.cleanup()
        if Path(f"{self.pg_path}/{self.port}.signal").exists():
            os.remove(f"{self.pg_path}/{self.port}.signal")

    def save_checkpoint(self, checkpoint_dir):
        # We can't actually do anything about this right now.
        pass

    def load_checkpoint(self, checkpoint_dir):
        # We can't actually do anything about this right now.
        pass
