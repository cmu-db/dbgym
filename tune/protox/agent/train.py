import json
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
import time
import click
import random
import logging

import ray
from ray.tune import TuneConfig
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune import Trainable
from ray.train import SyncConfig
from ray.air import RunConfig, FailureConfig

from tune.protox.agent.hpo import construct_wolp_config
from misc.utils import restart_ray, conv_inputpath_to_abspath, open_and_save, DEFAULT_PROTOX_CONFIG_RELPATH, default_benchmark_config_relpath, default_benchbase_config_relpath, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER, default_hpoed_agent_params_path, WORKSPACE_PATH_PLACEHOLDER, default_pgdata_snapshot_path, DEFAULT_WOLP_PARAMS_RELPATH, default_embedding_path, default_workload_path


class AgentTrainArgs:
    pass


@click.command()
@click.pass_obj
@click.argument("benchmark-name")
@click.argument("workload-name")
@click.option("--embedding-path", default=None, help=f"The path to the directory that contains an `embedding.pth` file with a trained encoder and decoder as well as a `config` file. The default is {default_embedding_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER)}")
@click.option(
    "--benchmark-config-path",
    default=None,
    type=Path,
    help=f"The path to the .yaml config file for the benchmark. The default is {default_benchmark_config_relpath(BENCHMARK_NAME_PLACEHOLDER)}.",
)
@click.option(
    "--benchbase-config-path",
    default=None,
    type=Path,
    help=f"The path to the .xml config file for BenchBase, used to run OLTP workloads. The default is {default_benchbase_config_relpath(BENCHMARK_NAME_PLACEHOLDER)}.",
)
@click.option("--protox-config-path", default=DEFAULT_PROTOX_CONFIG_RELPATH, help=f"The path to the file configuring lots of things about Proto-X.")
@click.option("--hpoed-agent-params-path", default=None, type=Path, help=f"The path to the agent params found by the HPO process. The default is {default_hpoed_agent_params_path(WORKSPACE_PATH_PLACEHOLDER)}.")
@click.option("--pgdata-snapshot-path", default=None, type=Path, help=f"The path to the .tgz snapshot of the pgdata directory for a specific workload. The default is {default_pgdata_snapshot_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER)}.")
@click.option("--agent-params-path", default=DEFAULT_WOLP_PARAMS_RELPATH, type=Path, help=f"The path to the parameters of the agent.")
@click.option(
    "--seed",
    default=None,
    type=int,
    help="The seed used for all sources of randomness (random, np, torch, etc.). The default is a random value.",
)
@click.option(
    "--workload-path",
    default=None,
    type=Path,
    help=f"The path to the directory that specifies the workload (such as its queries and order of execution). The default is {default_workload_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER)}.",
)
@click.option("--agent", default="wolp", help=f"The RL algorithm to use for the tuning agent.")
@click.option("--max-hpo-concurrent", default=1, help=f"The max # of concurrent agent models to train during hyperparameter optimization. This is usually set lower than `nproc` to reduce memory pressure.")
@click.option(
    "--num-samples",
    default=40,
    help=f"The # of times to specific hyperparameter configs to sample from the hyperparameter search space and train agent models with.",
)
@click.option("--early-kill", is_flag=True, help="Whether the tuner times out its steps.")
@click.option("--duration", default=0.01, type=float, help="The total number of hours to run for.")
@click.option("--workload-timeout", default=600, type=int, help="The timeout (in seconds) of a workload. We run the workload once per DBMS configuration. For OLAP workloads, certain configurations may be extremely suboptimal, so we need to time out the workload.")
@click.option("--query-timeout", default=30, type=int, help="The timeout (in seconds) of a query. See the help of --workload-timeout for the motivation of this.")
def train(dbgym_cfg, benchmark_name, workload_name, embedding_path, benchmark_config_path, benchbase_config_path, protox_config_path, hpoed_agent_params_path, pgdata_snapshot_path, agent_params_path, workload_path, seed, agent, max_hpo_concurrent, num_samples, early_kill, duration, workload_timeout, query_timeout):
    logging.info("agent.train(): called")
    
    # Set args to defaults programmatically (do this before doing anything else in the function)
    # TODO(phw2): figure out whether different scale factors use the same config
    # TODO(phw2): figure out what parts of the config should be taken out (like stuff about tables)
    if embedding_path == None:
        embedding_path = default_embedding_path(dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name)
    if benchmark_config_path == None:
        benchmark_config_path = default_benchmark_config_relpath(benchmark_name)
    if benchbase_config_path == None:
        benchbase_config_path = default_benchbase_config_relpath(benchmark_name)
    if hpoed_agent_params_path == None:
        hpoed_agent_params_path = default_hpoed_agent_params_path(dbgym_cfg.dbgym_workspace_path)
    if pgdata_snapshot_path == None:
        pgdata_snapshot_path = default_pgdata_snapshot_path(dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name)
    if workload_path == None:
        workload_path = default_workload_path(dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name)
    if seed == None:
        seed = random.randint(0, 1e8)

    # Convert all input paths to absolute paths
    embedding_path = conv_inputpath_to_abspath(dbgym_cfg, embedding_path)
    benchmark_config_path = conv_inputpath_to_abspath(dbgym_cfg, benchmark_config_path)
    benchbase_config_path = conv_inputpath_to_abspath(dbgym_cfg, benchbase_config_path)
    protox_config_path = conv_inputpath_to_abspath(dbgym_cfg, protox_config_path)
    hpoed_agent_params_path = conv_inputpath_to_abspath(dbgym_cfg, hpoed_agent_params_path)
    pgdata_snapshot_path = conv_inputpath_to_abspath(dbgym_cfg, pgdata_snapshot_path)
    agent_params_path = conv_inputpath_to_abspath(dbgym_cfg, agent_params_path)
    workload_path = conv_inputpath_to_abspath(dbgym_cfg, workload_path)

    # Build "args" object. TODO(phw2): after setting up E2E testing, including with agent HPO, refactor so we don't need the "args" object
    args = AgentTrainArgs()
    args.benchmark_name = benchmark_name
    args.workload_name = workload_name
    args.embedding_path = embedding_path
    args.benchmark_config_path = benchmark_config_path
    args.benchbase_config_path = benchbase_config_path
    args.protox_config_path = protox_config_path
    args.hpoed_agent_params_path = hpoed_agent_params_path
    args.pgdata_snapshot_path = pgdata_snapshot_path
    args.agent_params_path = agent_params_path
    args.workload_path = workload_path
    args.seed = seed
    args.agent = agent
    args.max_hpo_concurrent = max_hpo_concurrent
    args.num_samples = num_samples
    args.early_kill = early_kill
    args.duration = duration
    args.workload_timeout = workload_timeout
    args.query_timeout = query_timeout
    args = DotDict(args.__dict__)

    # Get the system knobs.
    with open_and_save(dbgym_cfg, protox_config_path, "r") as f:
        protox_config = yaml.safe_load(f)["protox"]
        system_knobs = protox_config["system_knobs"]

    # Per query knobs.
    with open_and_save(dbgym_cfg, benchmark_config_path, "r") as f:
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
    restart_ray()
    ray.init(address="localhost:6379", log_to_driver=False)

    # Config.
    if agent == "wolp":
        config = construct_wolp_config(dict(args))
    else:
        assert False, f"Unspecified agent {agent}"

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

    
    # Pass some extra needed stuff into config as well since there's no other way to get data to TuneOpt.setup()
    config["dbgym_cfg"] = dbgym_cfg
    config["is_oltp"] = is_oltp


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
        dbgym_cfg = hpo_config["dbgym_cfg"]
        protox_dir = dbgym_cfg.dbgym_repo_path
        # sys.path.append() must take in strings as input, not Path objects
        sys.path.append(str(protox_dir))

        from tune.protox.agent.tune_trial import TuneTrial, TimeoutChecker
        from tune.protox.agent.hpo import mutate_wolp_config
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
        self.timeout_checker = TimeoutChecker(protox_args["duration"])
        if protox_args.agent == "wolp":
            benchmark_name, pg_path, port = mutate_wolp_config(dbgym_cfg, self.logdir, hpo_config, protox_args)
        else:
            assert False, f"Unspecified agent {protox_args.agent}"

        self.pg_path = pg_path
        self.port = port

        # We will now overwrite the config files.
        protox_args["protox_config_path"] = Path(self.logdir) / "config.yaml"
        protox_args["agent_params_path"] = Path(self.logdir) / "model_params.yaml"
        protox_args["benchmark_config_path"] = Path(self.logdir) / f"{benchmark_name}.yaml"
        protox_args["reward"] = hpo_config.reward
        protox_args["horizon"] = hpo_config.horizon
        self.trial = TuneTrial()
        self.trial.setup(dbgym_cfg, hpo_config.is_oltp, protox_args, self.timeout_checker)
        self.start_time = time.time()

    def step(self):
        self.timeout_checker.resume()
        data = self.trial.step()

        # Decrement remaining time.
        self.timeout_checker.pause()
        if self.timeout_checker():
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