import json
import logging
import random
from datetime import datetime
from pathlib import Path

import click
import ray
import yaml
from ray.air import FailureConfig, RunConfig
from ray.train import SyncConfig
from ray.tune import TuneConfig
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator

from misc.utils import (
    BENCHMARK_NAME_PLACEHOLDER,
    DEFAULT_SYSKNOBS_RELPATH,
    WORKLOAD_NAME_PLACEHOLDER,
    WORKSPACE_PATH_PLACEHOLDER,
    SCALE_FACTOR_PLACEHOLDER,
    conv_inputpath_to_abspath,
    default_benchbase_config_path,
    default_benchmark_config_path,
    default_embedding_path,
    default_hpoed_agent_params_path,
    default_pristine_pgdata_snapshot_path,
    default_workload_path,
    open_and_save,
    restart_ray,
)
from tune.protox.agent.hpo import create_tune_opt_class


class AgentTrainArgs:
    pass


@click.command()
@click.pass_obj
@click.argument("benchmark-name")
@click.argument("workload-name")
@click.option(
    "--embedding-path",
    default=None,
    help=f"The path to the directory that contains an `embedding.pth` file with a trained encoder and decoder as well as a `config` file. The default is {default_embedding_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER, SCALE_FACTOR_PLACEHOLDER)}",
)
@click.option(
    "--benchmark-config-path",
    default=None,
    type=Path,
    help=f"The path to the .yaml config file for the benchmark. The default is {default_benchmark_config_path(BENCHMARK_NAME_PLACEHOLDER)}.",
)
@click.option(
    "--benchbase-config-path",
    default=None,
    type=Path,
    help=f"The path to the .xml config file for BenchBase, used to run OLTP workloads. The default is {default_benchbase_config_path(BENCHMARK_NAME_PLACEHOLDER)}.",
)
@click.option(
    "--sysknobs-path",
    default=DEFAULT_SYSKNOBS_RELPATH,
    help=f"The path to the file configuring the space of system knobs the tuner can tune.",
)
@click.option(
    "--hpoed-agent-params-path",
    default=None,
    type=Path,
    help=f"The path to the agent params found by the HPO process. The default is {default_hpoed_agent_params_path(WORKSPACE_PATH_PLACEHOLDER)}.",
)
@click.option(
    "--pristine-pgdata-snapshot-path",
    default=None,
    type=Path,
    help=f"The path to the .tgz snapshot of the pgdata directory for a specific workload. The default is {default_pristine_pgdata_snapshot_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, SCALE_FACTOR_PLACEHOLDER)}.",
)
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
@click.option(
    "--agent", default="wolp", help=f"The RL algorithm to use for the tuning agent."
)
@click.option(
    "--max-concurrent",
    default=1,
    help=f"The max # of concurrent agent models to train. Note that unlike in HPO, all will use the same hyperparameters. This just helps control for other sources of randomness.",
)
@click.option(
    "--num-samples",
    default=40,
    help=f"The # of times to specific hyperparameter configs to sample from the hyperparameter search space and train agent models with.",
)
@click.option(
    "--early-kill", is_flag=True, help="Whether the tuner times out its steps."
)
@click.option(
    "--duration", default=0.01, type=float, help="The total number of hours to run for."
)
@click.option(
    "--workload-timeout",
    default=600,
    type=int,
    help="The timeout (in seconds) of a workload. We run the workload once per DBMS configuration. For OLAP workloads, certain configurations may be extremely suboptimal, so we need to time out the workload.",
)
@click.option(
    "--query-timeout",
    default=30,
    type=int,
    help="The timeout (in seconds) of a query. See the help of --workload-timeout for the motivation of this.",
)
def tune(
    dbgym_cfg,
    benchmark_name,
    workload_name,
    embedding_path,
    benchmark_config_path,
    benchbase_config_path,
    sysknobs_path,
    hpoed_agent_params_path,
    pristine_pgdata_snapshot_path,
    workload_path,
    seed,
    agent,
    max_concurrent,
    num_samples,
    early_kill,
    duration,
    workload_timeout,
    query_timeout,
):
    logging.info("agent.train(): called")

    # Set args to defaults programmatically (do this before doing anything else in the function)
    # TODO(phw2): figure out whether different scale factors use the same config
    # TODO(phw2): figure out what parts of the config should be taken out (like stuff about tables)
    if embedding_path == None:
        embedding_path = default_embedding_path(
            dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name
        )
    if benchmark_config_path == None:
        benchmark_config_path = default_benchmark_config_path(benchmark_name)
    if benchbase_config_path == None:
        benchbase_config_path = default_benchbase_config_path(benchmark_name)
    if hpoed_agent_params_path == None:
        hpoed_agent_params_path = default_hpoed_agent_params_path(
            dbgym_cfg.dbgym_workspace_path
        )
    if pristine_pgdata_snapshot_path == None:
        pristine_pgdata_snapshot_path = default_pristine_pgdata_snapshot_path(
            dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name
        )
    if workload_path == None:
        workload_path = default_workload_path(
            dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name
        )
    if seed == None:
        seed = random.randint(0, 1e8)

    # Convert all input paths to absolute paths
    embedding_path = conv_inputpath_to_abspath(dbgym_cfg, embedding_path)
    benchmark_config_path = conv_inputpath_to_abspath(dbgym_cfg, benchmark_config_path)
    benchbase_config_path = conv_inputpath_to_abspath(dbgym_cfg, benchbase_config_path)
    sysknobs_path = conv_inputpath_to_abspath(dbgym_cfg, sysknobs_path)
    hpoed_agent_params_path = conv_inputpath_to_abspath(
        dbgym_cfg, hpoed_agent_params_path
    )
    pristine_pgdata_snapshot_path = conv_inputpath_to_abspath(dbgym_cfg, pristine_pgdata_snapshot_path)
    workload_path = conv_inputpath_to_abspath(dbgym_cfg, workload_path)

    # Build "args" object. TODO(phw2): after setting up E2E testing, including with agent HPO, refactor so we don't need the "args" object
    args = AgentTrainArgs()
    args.benchmark_name = benchmark_name
    args.workload_name = workload_name
    args.embedding_path = embedding_path
    args.benchmark_config_path = benchmark_config_path
    args.benchbase_config_path = benchbase_config_path
    args.sysknobs_path = sysknobs_path
    args.hpoed_agent_params_path = hpoed_agent_params_path
    args.pristine_pgdata_snapshot_path = pristine_pgdata_snapshot_path
    args.workload_path = workload_path
    args.seed = seed
    args.agent = agent
    args.max_concurrent = max_concurrent
    args.num_samples = num_samples
    args.early_kill = early_kill
    args.duration = duration
    args.workload_timeout = workload_timeout
    args.query_timeout = query_timeout
    args = DotDict(args.__dict__)

    # Get the system knobs.
    with open_and_save(dbgym_cfg, sysknobs_path, "r") as f:
        system_knobs = yaml.safe_load(f)["system_knobs"]

    # Per query knobs.
    with open_and_save(dbgym_cfg, benchmark_config_path, "r") as f:
        benchmark_config = yaml.safe_load(f)["protox"]
        per_query_scan_method = benchmark_config["per_query_scan_method"]
        per_query_select_parallel = benchmark_config["per_query_select_parallel"]
        index_space_aux_type = benchmark_config["index_space_aux_type"]
        index_space_aux_include = benchmark_config["index_space_aux_include"]
        per_query_knobs = benchmark_config["per_query_knobs"]
        per_query_knob_gen = benchmark_config["per_query_knob_gen"]
        query_spec = benchmark_config["query_spec"]
        is_oltp = benchmark_config["oltp_workload"]

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

    # Pass other configs through
    config["is_oltp"] = is_oltp

    # Scheduler.
    scheduler = FIFOScheduler()

    hpoed_agent_params = None
    print(f"hpoed_agent_params_path={hpoed_agent_params_path}")
    if hpoed_agent_params_path is not None:
        with open_and_save(dbgym_cfg, hpoed_agent_params_path, "r") as f:
            hpoed_agent_params = json.load(f)

            for config in hpoed_agent_params:
                if "protox_per_query_knobs" not in config:
                    config["protox_per_query_knobs"] = per_query_knobs
                if "protox_per_query_scan_method" not in config:
                    config["protox_per_query_scan_method"] = per_query_scan_method
                if "protox_per_query_select_parallel" not in config:
                    config["protox_per_query_select_parallel"] = (
                        per_query_select_parallel
                    )
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

    # Search.
    # if hpoed_agent_params == None, hyperparameter optimization will be performend
    # if hpoed_agent_params != None, we will just run a single tuning job with the params hpoed_agent_params
    search = BasicVariantGenerator(
        points_to_evaluate=hpoed_agent_params, max_concurrent=max_concurrent
    )

    # for OLTP, we're trying to *maximize* txn / sec. for OLAP, we're trying to *minimize* total runtime
    mode = "max" if is_oltp else "min"
    tune_config = TuneConfig(
        scheduler=scheduler,
        search_alg=search,
        num_samples=num_samples,
        max_concurrent_trials=max_concurrent,
        chdir_to_trial_dir=True,
        metric=METRIC,
        mode=mode,
    )

    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_config = RunConfig(
        name=f"ProtoxHPO_{dtime}",
        failure_config=FailureConfig(max_failures=0, fail_fast=True),
        sync_config=SyncConfig(),
        verbose=2,
        log_to_file=True,
    )

    tuner = ray.tune.Tuner(
        create_tune_opt_class(dbgym_cfg),
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
    print(
        "Best hyperparameters found were: ",
        results.get_best_result(metric=METRIC, mode="max").config,
    )


METRIC = "Best Metric"


class DotDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
