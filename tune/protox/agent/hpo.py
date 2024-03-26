import sys
import time
import json
import yaml
from pathlib import Path
from ray import tune
import numpy as np
import torch
import os
import pandas as pd
from datetime import datetime
from typing import Any, Union
import random
import click
import ray
from ray.tune import Trainable
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune import TuneConfig
from ray.air import RunConfig, FailureConfig
from ray.train import SyncConfig

from tune.protox.agent.coerce_params import coerce_params
from tune.protox.agent.build_trial import build_trial
from misc.utils import DBGymConfig, open_and_save, restart_ray, conv_inputpath_to_realabspath, default_pristine_pgdata_snapshot_path, default_workload_path, default_embedding_path, default_benchmark_config_path, default_benchbase_config_path, WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER, SCALE_FACTOR_PLACEHOLDER, DEFAULT_SYSKNOBS_RELPATH, default_pgbin_path


class AgentHPOArgs:
    def __init__(self, benchmark_name, workload_name, embedding_path, benchmark_config_path, benchbase_config_path, sysknobs_path, pristine_pgdata_snapshot_path, pgbin_path, workload_path, seed, agent, max_concurrent, num_samples, early_kill, duration, workload_timeout, query_timeout):
        self.benchmark_name = benchmark_name
        self.workload_name = workload_name
        self.embedding_path = embedding_path
        self.benchmark_config_path = benchmark_config_path
        self.benchbase_config_path = benchbase_config_path
        self.sysknobs_path = sysknobs_path
        self.pristine_pgdata_snapshot_path = pristine_pgdata_snapshot_path
        self.pgbin_path = pgbin_path
        self.workload_path = workload_path
        self.seed = seed
        self.agent = agent
        self.max_concurrent = max_concurrent
        self.num_samples = num_samples
        self.early_kill = early_kill
        self.duration = duration
        self.workload_timeout = workload_timeout
        self.query_timeout = query_timeout


@click.command()
@click.pass_obj
@click.argument("benchmark-name")
@click.argument("workload-name")
@click.option(
    "--scale-factor",
    default=1.0,
    help=f"The scale factor used when generating the data of the benchmark.",
)
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
    "--pristine-pgdata-snapshot-path",
    default=None,
    type=Path,
    help=f"The path to the .tgz snapshot of the pgdata directory to use as a starting point for tuning. The default is {default_pristine_pgdata_snapshot_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, SCALE_FACTOR_PLACEHOLDER)}.",
)
@click.option(
    "--pgbin-path",
    default=None,
    type=Path,
    help=f"The path to the bin containing Postgres executables. The default is {default_pgbin_path(WORKSPACE_PATH_PLACEHOLDER)}.",
)
@click.option(
    "--workload-path",
    default=None,
    type=Path,
    help=f"The path to the directory that specifies the workload (such as its queries and order of execution). The default is {default_workload_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER)}.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="The seed used for all sources of randomness (random, np, torch, etc.). The default is a random value.",
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
def hpo(
    dbgym_cfg,
    benchmark_name,
    workload_name,
    scale_factor,
    embedding_path,
    benchmark_config_path,
    benchbase_config_path,
    sysknobs_path,
    pristine_pgdata_snapshot_path,
    pgbin_path,
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
    # Set args to defaults programmatically (do this before doing anything else in the function)
    if embedding_path == None:
        embedding_path = default_embedding_path(dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name, scale_factor)
    if benchmark_config_path == None:
        benchmark_config_path = default_benchmark_config_path(benchmark_name)
    if benchbase_config_path == None:
        benchbase_config_path = default_benchbase_config_path(benchmark_name)
    if pristine_pgdata_snapshot_path == None:
        pristine_pgdata_snapshot_path = default_pristine_pgdata_snapshot_path(dbgym_cfg.dbgym_workspace_path, benchmark_name, scale_factor)
    if pgbin_path == None:
        pgbin_path = default_pgbin_path(dbgym_cfg.dbgym_workspace_path)
    if workload_path == None:
        workload_path = default_workload_path(dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name)
    if seed == None:
        seed = random.randint(0, 1e8)

    # Convert all input paths to absolute paths
    embedding_path = conv_inputpath_to_realabspath(dbgym_cfg, embedding_path)
    benchmark_config_path = conv_inputpath_to_realabspath(dbgym_cfg, benchmark_config_path)
    benchbase_config_path = conv_inputpath_to_realabspath(dbgym_cfg, benchbase_config_path)
    sysknobs_path = conv_inputpath_to_realabspath(dbgym_cfg, sysknobs_path)
    pristine_pgdata_snapshot_path = conv_inputpath_to_realabspath(dbgym_cfg, pristine_pgdata_snapshot_path)
    pgbin_path = conv_inputpath_to_realabspath(dbgym_cfg, pgbin_path)
    workload_path = conv_inputpath_to_realabspath(dbgym_cfg, workload_path)

    # Create args object
    hpo_args = AgentHPOArgs(benchmark_name, workload_name, embedding_path, benchmark_config_path, benchbase_config_path, sysknobs_path, pristine_pgdata_snapshot_path, pgbin_path, workload_path, seed, agent, max_concurrent, num_samples, early_kill, duration, workload_timeout, query_timeout)
    _tune_hpo(dbgym_cfg, hpo_args)


# The reason we put the paths inside the space is so that the tuner only receives the space .json file
#   as a CLI arg and doesn't need any other CLI args. The hyperparameters are selected using the paths
#   given here, so it doesn't make sense to specify them separately when tuning.
def _build_space(
    sysknobs: dict[str, Any],
    benchmark_config: dict[str, Any],
    pristine_pgdata_snapshot_path: Path,
    workload_path: Path,
    embedding_path: list[str],
    pgconn_info: dict[str, str],
    benchbase_config: dict[str, Any]={},
    duration: int=30,
    seed: int=0,
    workload_timeouts: list[int]=[600],
    query_timeouts: list[int]=[30],
) -> dict[str, Any]:

    return {
        # Internal space versioning.
        "space_version": "2.0",
        "verbose": True,
        "trace": True,
        "seed": seed,
        # Timeouts.
        "duration": duration,
        "workload_timeout": tune.choice(workload_timeouts),
        "query_timeout": tune.choice(query_timeouts),
        # Paths.
        "pristine_pgdata_snapshot_path": str(pristine_pgdata_snapshot_path),
        "workload_path": str(workload_path),
        "output_log_path": "artifacts/",
        "pgconn_info": pgconn_info,
        "benchmark_config": benchmark_config,
        "benchbase_config": benchbase_config,
        # Horizon before resetting.
        "horizon": 5,
        # Workload Eval.
        "workload_eval_mode": tune.choice(["global_dual", "prev_dual", "all"]),
        "workload_eval_inverse": tune.choice([False, True]),
        "workload_eval_reset": tune.choice([False, True]),
        # Reward.
        "reward": tune.choice(["multiplier", "relative", "cdb_delta"]),
        "reward_scaler": tune.choice([1, 2, 5, 10]),
        "workload_timeout_penalty": tune.choice([1, 2, 4]),
        # State.
        "metric_state": tune.choice(["metric", "structure", "structure_normalize"]),
        "maximize_state": not benchmark_config.get("oltp_workload", False),
        # Whether to normalize state or not.
        "normalize_state": tune.sample_from(
            lambda spc: False
            if spc["config"]["metric_state"] == "structure_normalize"
            else bool(np.random.choice([False, True]))
        ),
        # Whether to normalize reward or not.
        "normalize_reward": tune.choice([False, True]),
        # Default quantization factor to use.
        "default_quantization_factor": 100,
        "system_knobs": sysknobs,
        # Embeddings.
        "embedding_path": tune.choice(embedding_path),
        # LSC Parameters.
        # Note that the units for these are based on the embedding itself.
        "lsc": {
            "enabled": False,
            # These are the initial low-bias, comma separated by the horizon step.
            "initial": "0",
            # These are the units for how much to increment the low-bias by each time.
            "increment": "0",
            # Maximum allowed shift.
            "max": "0",
            # This controls how frequently to try and boost the shifts based on episode.
            "shift_eps_freq": 1,
            # How many episodes to start.
            "shift_after": 3,
        },
        # RL Agent Parameters.
        # Number of warmup steps.
        "learning_starts": 0,
        # Learning rate.
        "learning_rate": tune.choice([1e-3, 8e-4, 6e-4, 3e-4, 5e-5, 3e-5, 1e-5]),
        "critic_lr_scale": tune.choice([1.0, 2.5, 5.0, 7.5, 10.0]),
        "policy_l2_reg": tune.choice([0.0, 0.01, 0.03, 0.05]),
        # Discount.
        "gamma": tune.choice([0, 0.9, 0.95, 0.995, 1.0]),
        # Polyak averaging rate.
        "tau": tune.choice([1.0, 0.99, 0.995]),
        # Replay Buffer Size.
        "buffer_size": 1_000_000,
        # Batch size.
        "batch_size": tune.choice([8, 16, 32, 64]),
        # Gradient Clipping.
        "grad_clip": tune.choice([1.0, 5.0, 10.0]),
        # Gradient steps per sample.
        "gradient_steps": tune.choice([1, 2, 4]),
        # Target noise.
        "target_noise": {
            "target_noise_clip": tune.choice([0, 0.05, 0.1, 0.15]),
            "target_policy_noise": tune.sample_from(
                lambda spc: 0.1
                if spc["config"]["target_noise"]["target_noise_clip"] == 0
                else float(np.random.choice([0.05, 0.1, 0.15, 0.2]))
            ),
        },
        # Training steps.
        "train_freq_unit": tune.choice(["step", "episode"]),
        "train_freq_frequency": tune.sample_from(
            lambda spc: 1
            if spc["config"]["train_freq_unit"] == "episode"
            else int(np.random.choice([1, 2]))
        ),
        # Noise parameters.
        "noise_parameters": {
            "noise_type": tune.choice(["normal", "ou"]),
            "noise_sigma": tune.choice([0.01, 0.05, 0.1, 0.15, 0.2]),
        },
        "scale_noise_perturb": True,
        # Neighbor parameters.
        "neighbor_parameters": {
            "knob_num_nearest": tune.choice([100, 200]),
            "knob_span": tune.choice([1, 2]),
            "index_num_samples": 1,
            "index_rules": tune.choice([False, True]),
        },
        # Networks.
        "weight_init": tune.choice(["xavier_normal", "xavier_uniform", "orthogonal"]),
        "bias_zero": tune.choice([False, True]),
        "policy_weight_adjustment": tune.choice([1, 100]),
        "activation_fn": tune.choice(["gelu", "mish"]),
        "pi_arch": tune.choice(["128", "256", "128,128", "256,256", "512", "256,512"]),
        "qf_arch": tune.choice(
            [
                "256,64",
                "256,256",
                "256,128,128",
                "256,64,64",
                "512",
                "512,256",
                "1024",
                "1024,256",
            ]
        ),
    }


class TuneTimeoutChecker(object):
    def __init__(self, duration: int) -> None:
        self.limit = (duration * 3600) > 0
        self.remain = int(duration * 3600)
        self.running = False
        self.start = 0.

    def resume(self) -> None:
        self.start = time.time()
        self.running = True

    def pause(self) -> None:
        if self.limit and self.running:
            self.remain -= int(time.time() - self.start)
        self.running = False

    def __call__(self) -> bool:
        if not self.limit:
            return False

        if self.remain <= 0:
            return True

        if self.running:
            return int(time.time() - self.start) >= self.remain

        return False


class TuneTrial:
    def __init__(self, dbgym_cfg: DBGymConfig) -> None:
        self.dbgym_cfg = dbgym_cfg

    def setup(self, hpo_config: dict[str, Any]) -> None:
        # Attach mythril directory to the search path.
        sys.path.append(os.path.expanduser(self.dbgym_cfg.dbgym_repo_path))

        torch.set_default_dtype(torch.float32) # type: ignore
        seed = (
            hpo_config["seed"]
            if hpo_config["seed"] != -1
            else np.random.randint(np.iinfo(np.int32).max)
        )
        np.random.seed(seed)
        torch.manual_seed(seed)
        assert hasattr(self, "logdir")

        self.timeout = TuneTimeoutChecker(hpo_config["duration"])
        self.logger, self.target_reset, self.env, self.agent, self.signal = build_trial(
            self.dbgym_cfg,
            seed=seed,
            logdir=self.logdir,
            hpo_config=hpo_config
        )
        self.logger.get_logger(None).info("%s", hpo_config)
        self.logger.get_logger(None).info(f"Seed: {seed}")

        # Attach the timeout checker and loggers.
        self.agent.set_timeout_checker(self.timeout)
        self.agent.set_logger(self.logger)

        self.env_init = False
        self.start_time = time.time()
        self.step_count = 0

    def step(self) -> dict[Any, Any]:
        self.step_count += 1
        # Only measure the actual tuning time.
        self.timeout.resume()

        episode = self.agent._episode_num
        it = self.agent.num_timesteps
        self.logger.get_logger(None).info(
            f"Starting episode: {episode+1}, iteration: {it+1}"
        )

        if not self.env_init:
            _, infos = self.env.reset()
            baseline_reward, baseline_metric = (
                infos["baseline_reward"],
                infos["baseline_metric"],
            )
            self.logger.get_logger(None).info(
                f"Baseline Metric: {baseline_metric}. Baseline Reward: {baseline_reward}"
            )
            self.env_init = True
            self.logger.stash_results(infos, name_override="baseline")
        else:
            self.agent.learn(self.env, total_timesteps=1)

        self.timeout.pause()
        self.logger.advance()

        # Step telemetry that we care about.
        data = {
            "AgentEpisode": episode,
            "AgentTimesteps": it,
            "TrialStep": self.step_count,
            "Best Metric": self.target_reset.real_best_metric
            if self.target_reset
            else -1,
            "Best Seen Metric": self.target_reset.best_metric
            if self.target_reset
            else -1,
            "HoursElapsed": (time.time() - self.start_time) / 3600.,
        }

        # If we've timed out. Note that we've timed out.
        if self.timeout():
            self.cleanup()
            data[ray.tune.result.DONE] = True

        return data

    def cleanup(self) -> None:
        self.logger.flush()
        self.env.close() # type: ignore
        if Path(self.signal).exists():
            os.remove(self.signal)

# I want to pass dbgym_cfg into TuneOpt without putting it inside `hpo_config`. This is because it's a pain to turn DBGymConfig
#   into a nice dictionary of strings, and nothing in DBGymConfig would be relevant to someone checking the configs later
# Using a function to create a class is Ray's recommended way of doing this (see
#   https://discuss.ray.io/t/using-static-variables-to-control-trainable-subclass-in-ray-tune/808/4)
# If you don't create the class with a function, it doesn't work due to how Ray serializes classes
def create_tune_opt_class(dbgym_cfg_param):
    global global_dbgym_cfg
    global_dbgym_cfg = dbgym_cfg_param

    class TuneOpt(Trainable):
        dbgym_cfg = global_dbgym_cfg

        def setup(self, hpo_config: dict[str, Any]) -> None:
            self.trial = TuneTrial(TuneOpt.dbgym_cfg)
            self.trial.logdir = self.logdir # type: ignore
            self.trial.setup(hpo_config)

        def step(self) -> dict[Any, Any]:
            return self.trial.step()

        def cleanup(self) -> None:
            return self.trial.cleanup()

        def save_checkpoint(self, checkpoint_dir: str) -> None:
            # We can't actually do anything about this right now.
            pass

        def load_checkpoint(self, checkpoint_dir: Union[dict[Any, Any], None]) -> None:
            # We can't actually do anything about this right now.
            pass
    
    return TuneOpt


# This is used when you already have a good set of HPOs and just want to tune the DBMS
def tune_single_trial(args: Any) -> None:
    with open(args.hpo_params_file, "r") as f:
        hpo_config = json.load(f)

    # Coerce using a dummy space.
    hpo_config = coerce_params(_build_space(
        sysknobs={},
        benchmark_config={},
        data_snapshot="",
        embedding_path=[],
        pgconn_info={}
    ), hpo_config)

    # Assume we are executing from the root.
    # TODO(phw2): get this from dbgym_cfg
    hpo_config["dbgym_dir"] = os.getcwd()

    # Get the duration.
    assert "duration" in hpo_config

    # Piggyback off the HPO magic.
    t = TuneTrial()
    # This is a hack.
    t.logdir = Path("artifacts/") # type: ignore
    t.logdir.mkdir(parents=True, exist_ok=True) # type: ignore
    t.setup(hpo_config)
    start = time.time()

    data = []
    while (time.time() - start) < hpo_config["duration"] * 3600:
        data.append(t.step())

        # Continuously write the file out.
        pd.DataFrame(data).to_csv(args.output_step_data, index=False)

    t.cleanup()
    # Output the step data.
    pd.DataFrame(data).to_csv(args.output_step_data, index=False)


def _tune_hpo(dbgym_cfg: DBGymConfig, hpo_args: AgentHPOArgs) -> None:
    with open_and_save(dbgym_cfg, hpo_args.sysknobs_path) as f:
        sysknobs = yaml.safe_load(f)["system_knobs"]

    with open_and_save(dbgym_cfg, hpo_args.benchmark_config_path) as f:
        benchmark_config = yaml.safe_load(f)
        is_oltp = benchmark_config["protox"]["query_spec"]["oltp_workload"]
        benchmark = [k for k in benchmark_config.keys()][0]
        benchmark_config = benchmark_config[benchmark]
        benchmark_config["benchmark"] = benchmark

    # TODO(phw2): read the dir hpo_args.embedding_path and get a list of embeddings
    embedding_path = [hpo_args.embedding_path]
    # TODO(phw2): make workload and query timeout params lists instead of just ints
    workload_timeouts = [hpo_args.workload_timeout]
    query_timeouts = [hpo_args.query_timeout]

    benchbase_config = {
        "oltp_config": {
            "oltp_num_terminals": hpo_args.oltp_num_terminals,
            "oltp_duration": hpo_args.oltp_duration,
            "oltp_sf": hpo_args.oltp_sf,
            "oltp_warmup": hpo_args.oltp_warmup,
        },
        "benchbase_path": hpo_args.benchbase_path,
        "benchbase_config_path": hpo_args.benchbase_config_path,
    } if is_oltp else {}

    space = _build_space(
        sysknobs,
        benchmark_config,
        hpo_args.pristine_pgdata_snapshot_path,
        hpo_args.workload_path,
        embedding_path,
        pgconn_info={
            "pristine_pgdata_snapshot_path": hpo_args.pristine_pgdata_snapshot_path,
            "pgbin_path": hpo_args.pgbin_path,
        },
        benchbase_config=benchbase_config,
        duration=hpo_args.duration,
        seed=hpo_args.seed,
        workload_timeouts=workload_timeouts,
        query_timeouts=query_timeouts,
    )

    restart_ray()
    ray.init(address="localhost:6379", log_to_driver=False)

    # Scheduler.
    scheduler = FIFOScheduler() # type: ignore

    # Search.
    search = BasicVariantGenerator(
        max_concurrent=hpo_args.max_concurrent
    )

    tune_config = TuneConfig(
        scheduler=scheduler,
        search_alg=search,
        num_samples=hpo_args.num_samples,
        max_concurrent_trials=hpo_args.max_concurrent,
        chdir_to_trial_dir=True,
        metric="Best Metric",
        mode="max" if is_oltp else "min",
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
        create_tune_opt_class(dbgym_cfg),
        tune_config=tune_config,
        run_config=run_config,
        param_space=space,
    )

    results = tuner.fit()
    if results.num_errors > 0:
        for i in range(len(results)):
            if results[i].error:
                print(f"Trial {results[i]} FAILED")
        assert False, print("Encountered exceptions!")