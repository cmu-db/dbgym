import json
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Type, Union

import click
import numpy as np
import pandas as pd
import ray
import torch
import yaml
from ray import tune
from ray.air import FailureConfig, RunConfig
from ray.train import SyncConfig
from ray.tune import Trainable, TuneConfig
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator

from benchmark.constants import DEFAULT_SCALE_FACTOR
from tune.protox.agent.build_trial import build_trial
from util.log import DBGYM_LOGGER_NAME
from util.workspace import (
    BENCHMARK_NAME_PLACEHOLDER,
    DEFAULT_BOOT_CONFIG_FPATH,
    DEFAULT_SYSKNOBS_PATH,
    DEFAULT_WORKLOAD_TIMEOUT,
    SCALE_FACTOR_PLACEHOLDER,
    WORKLOAD_NAME_PLACEHOLDER,
    WORKSPACE_PATH_PLACEHOLDER,
    DBGymConfig,
    TuningMode,
    default_dbdata_parent_dpath,
    default_pgbin_path,
    default_pristine_dbdata_snapshot_path,
    fully_resolve_path,
    get_default_benchbase_config_path,
    get_default_benchmark_config_path,
    get_default_embedder_path,
    get_default_hpoed_agent_params_fname,
    get_default_workload_name_suffix,
    get_default_workload_path,
    get_workload_name,
    is_ssd,
    link_result,
    open_and_save,
    restart_ray,
)

METRIC_NAME = "Best Metric"


class AgentHPOArgs:
    def __init__(
        self,
        benchmark_name: str,
        workload_name: str,
        embedder_path: Path,
        benchmark_config_path: Path,
        benchbase_config_path: Path,
        sysknobs_path: Path,
        pristine_dbdata_snapshot_path: Path,
        dbdata_parent_dpath: Path,
        pgbin_path: Path,
        workload_path: Path,
        seed: int,
        agent: str,
        max_concurrent: int,
        num_samples: int,
        tune_duration_during_hpo: float,
        workload_timeout: float,
        query_timeout: float,
        enable_boot_during_hpo: bool,
        boot_config_fpath_during_hpo: Path,
        build_space_good_for_boot: bool,
    ):
        self.benchmark_name = benchmark_name
        self.workload_name = workload_name
        self.embedder_path = embedder_path
        self.benchmark_config_path = benchmark_config_path
        self.benchbase_config_path = benchbase_config_path
        self.sysknobs_path = sysknobs_path
        self.pristine_dbdata_snapshot_path = pristine_dbdata_snapshot_path
        self.dbdata_parent_dpath = dbdata_parent_dpath
        self.pgbin_path = pgbin_path
        self.workload_path = workload_path
        self.seed = seed
        self.agent = agent
        self.max_concurrent = max_concurrent
        self.num_samples = num_samples
        self.tune_duration_during_hpo = tune_duration_during_hpo
        self.workload_timeout = workload_timeout
        self.query_timeout = query_timeout
        self.enable_boot_during_hpo = enable_boot_during_hpo
        self.boot_config_fpath_during_hpo = boot_config_fpath_during_hpo
        self.build_space_good_for_boot = build_space_good_for_boot


@click.command()
@click.pass_obj
@click.argument("benchmark-name")
@click.option(
    "--workload-name-suffix",
    type=str,
    default=None,
    help=f"The suffix of the workload name (the part after the scale factor).",
)
@click.option(
    "--scale-factor",
    type=float,
    default=DEFAULT_SCALE_FACTOR,
    help=f"The scale factor used when generating the data of the benchmark.",
)
@click.option(
    "--embedder-path",
    type=Path,
    default=None,
    help=f"The path to the directory that contains an `embedder.pth` file with a trained encoder and decoder as well as a `config` file. The default is {get_default_embedder_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER)}",
)
@click.option(
    "--benchmark-config-path",
    type=Path,
    default=None,
    help=f"The path to the .yaml config file for the benchmark. The default is {get_default_benchmark_config_path(BENCHMARK_NAME_PLACEHOLDER)}.",
)
@click.option(
    "--benchbase-config-path",
    type=Path,
    default=None,
    help=f"The path to the .xml config file for BenchBase, used to run OLTP workloads. The default is {get_default_benchbase_config_path(BENCHMARK_NAME_PLACEHOLDER)}.",
)
@click.option(
    "--sysknobs-path",
    type=Path,
    default=DEFAULT_SYSKNOBS_PATH,
    help=f"The path to the file configuring the space of system knobs the tuner can tune.",
)
@click.option(
    "--pristine-dbdata-snapshot-path",
    type=Path,
    default=None,
    help=f"The path to the .tgz snapshot of the dbdata directory to use as a starting point for tuning. The default is {default_pristine_dbdata_snapshot_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, SCALE_FACTOR_PLACEHOLDER)}.",
)
@click.option(
    "--intended-dbdata-hardware",
    type=click.Choice(["hdd", "ssd"]),
    default="hdd",
    help=f"The intended hardware dbdata should be on. Used as a sanity check for --dbdata-parent-dpath.",
)
@click.option(
    "--dbdata-parent-dpath",
    type=Path,
    default=None,
    help=f"The path to the parent directory of the dbdata which will be actively tuned. The default is {default_dbdata_parent_dpath(WORKSPACE_PATH_PLACEHOLDER)}.",
)
@click.option(
    "--pgbin-path",
    type=Path,
    default=None,
    help=f"The path to the bin containing Postgres executables. The default is {default_pgbin_path(WORKSPACE_PATH_PLACEHOLDER)}.",
)
@click.option(
    "--workload-path",
    type=Path,
    default=None,
    help=f"The path to the directory that specifies the workload (such as its queries and order of execution). The default is {get_default_workload_path(WORKSPACE_PATH_PLACEHOLDER, BENCHMARK_NAME_PLACEHOLDER, WORKLOAD_NAME_PLACEHOLDER)}.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="The seed used for all sources of randomness (random, np, torch, etc.). The default is a random value.",
)
@click.option(
    "--agent",
    type=str,
    default="wolp",
    help=f"The RL algorithm to use for the tuning agent.",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=1,
    help=f"The max # of concurrent agent models to train. Note that unlike in HPO, all will use the same hyperparameters. This just helps control for other sources of randomness.",
)
@click.option(
    "--num-samples",
    type=int,
    default=40,
    help=f"The # of times to specific hyperparameter configs to sample from the hyperparameter search space and train agent models with.",
)
@click.option(
    "--tune-duration-during-hpo",
    type=float,
    default=4.0,
    help="The number of hours to run each hyperparamer config tuning trial for.",
)
@click.option(
    "--workload-timeout",
    type=int,
    default=DEFAULT_WORKLOAD_TIMEOUT,
    help="The timeout (in seconds) of a workload. We run the workload once per DBMS configuration. For OLAP workloads, certain configurations may be extremely suboptimal, so we need to time out the workload.",
)
@click.option(
    "--query-timeout",
    type=int,
    default=30,
    help="The timeout (in seconds) of a query. See the help of --workload-timeout for the motivation of this.",
)
@click.option(
    "--enable-boot-during-hpo",
    is_flag=True,
    help="Whether to enable the Boot query accelerator during the HPO process. Deciding to use Boot during HPO is separate from deciding to use Boot during tuning.",
)
@click.option(
    "--boot-config-fpath-during-hpo",
    type=Path,
    default=DEFAULT_BOOT_CONFIG_FPATH,
    help="The path to the file configuring Boot when running HPO. When tuning, you may use a different Boot config.",
)
# Building a space good for Boot is subtly different from whether we enable Boot during HPO.
# There are certain options that qualitatively do not perform well with Boot (e.g. metrics state
#   because Boot extrapolates the query runtime but not metrics). This param controls whether we
#   use those options or not.
# I chose the word "good" instead of "compatible" because metrics state does not _crash_ if you
#   use Boot but it just doesn't seem like it would perform well.
# One workflow where these two variables are different is where we don't enable Boot during HPO
#   but do want to enable Boot during tuning.
# However, whether we're building a space good for Boot is also different from whether we enable
#   Boot during tuning. We often want to compare one tuning run with Boot against one without
#   Boot, in which case we'd build a space good for Boot and then run it once with Boot and once
#   without Boot.
@click.option(
    "--build-space-good-for-boot",
    is_flag=True,
    help="Whether to avoid certain options that are known to not perform well when Boot is enabled. See the codebase for why this is different from --enable-boot-during-hpo.",
)
def hpo(
    dbgym_cfg: DBGymConfig,
    benchmark_name: str,
    workload_name_suffix: Optional[str],
    scale_factor: float,
    embedder_path: Optional[Path],
    benchmark_config_path: Optional[Path],
    benchbase_config_path: Optional[Path],
    sysknobs_path: Path,
    pristine_dbdata_snapshot_path: Optional[Path],
    intended_dbdata_hardware: str,
    dbdata_parent_dpath: Optional[Path],
    pgbin_path: Optional[Path],
    workload_path: Optional[Path],
    seed: Optional[int],
    agent: str,
    max_concurrent: int,
    num_samples: int,
    tune_duration_during_hpo: float,
    workload_timeout: int,
    query_timeout: int,
    enable_boot_during_hpo: bool,
    boot_config_fpath_during_hpo: Path,
    build_space_good_for_boot: bool,
) -> None:
    # Set args to defaults programmatically (do this before doing anything else in the function)
    if workload_name_suffix is None:
        workload_name_suffix = get_default_workload_name_suffix(benchmark_name)
    workload_name = get_workload_name(scale_factor, workload_name_suffix)
    if embedder_path is None:
        embedder_path = get_default_embedder_path(
            dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name
        )
    if benchmark_config_path is None:
        benchmark_config_path = get_default_benchmark_config_path(benchmark_name)
    if benchbase_config_path is None:
        benchbase_config_path = get_default_benchbase_config_path(benchmark_name)
    if pristine_dbdata_snapshot_path is None:
        pristine_dbdata_snapshot_path = default_pristine_dbdata_snapshot_path(
            dbgym_cfg.dbgym_workspace_path, benchmark_name, scale_factor
        )
    if dbdata_parent_dpath is None:
        dbdata_parent_dpath = default_dbdata_parent_dpath(
            dbgym_cfg.dbgym_workspace_path
        )
    if pgbin_path is None:
        pgbin_path = default_pgbin_path(dbgym_cfg.dbgym_workspace_path)
    if workload_path is None:
        workload_path = get_default_workload_path(
            dbgym_cfg.dbgym_workspace_path, benchmark_name, workload_name
        )
    if seed is None:
        seed = random.randint(0, int(1e8))

    # Fully resolve all input paths.
    embedder_path = fully_resolve_path(dbgym_cfg, embedder_path)
    benchmark_config_path = fully_resolve_path(dbgym_cfg, benchmark_config_path)
    benchbase_config_path = fully_resolve_path(dbgym_cfg, benchbase_config_path)
    sysknobs_path = fully_resolve_path(dbgym_cfg, sysknobs_path)
    pristine_dbdata_snapshot_path = fully_resolve_path(
        dbgym_cfg, pristine_dbdata_snapshot_path
    )
    dbdata_parent_dpath = fully_resolve_path(dbgym_cfg, dbdata_parent_dpath)
    pgbin_path = fully_resolve_path(dbgym_cfg, pgbin_path)
    workload_path = fully_resolve_path(dbgym_cfg, workload_path)
    boot_config_fpath_during_hpo = fully_resolve_path(
        dbgym_cfg, boot_config_fpath_during_hpo
    )

    # Check assertions on args
    if intended_dbdata_hardware == "hdd":
        assert not is_ssd(
            dbdata_parent_dpath
        ), f"Intended hardware is HDD but dbdata_parent_dpath ({dbdata_parent_dpath}) is an SSD"
    elif intended_dbdata_hardware == "ssd":
        assert is_ssd(
            dbdata_parent_dpath
        ), f"Intended hardware is SSD but dbdata_parent_dpath ({dbdata_parent_dpath}) is an HDD"
    else:
        assert False

    # Create args object
    hpo_args = AgentHPOArgs(
        benchmark_name,
        workload_name,
        embedder_path,
        benchmark_config_path,
        benchbase_config_path,
        sysknobs_path,
        pristine_dbdata_snapshot_path,
        dbdata_parent_dpath,
        pgbin_path,
        workload_path,
        seed,
        agent,
        max_concurrent,
        num_samples,
        tune_duration_during_hpo,
        workload_timeout,
        query_timeout,
        enable_boot_during_hpo,
        boot_config_fpath_during_hpo,
        build_space_good_for_boot,
    )
    _tune_hpo(dbgym_cfg, hpo_args)


# The reason we put the paths inside the space is so that the tuner only receives the space .json file
#   as a CLI arg and doesn't need any other CLI args. The hyperparameters are selected using the paths
#   given here, so it doesn't make sense to specify them separately when tuning.
def build_space(
    sysknobs: dict[str, Any],
    benchmark_config: dict[str, Any],
    workload_path: Path,
    embedder_path: list[Path],
    pgconn_info: dict[str, Path],
    benchbase_config: dict[str, Any] = {},
    tune_duration_during_hpo: float = 30.0,
    seed: int = 0,
    enable_boot_during_hpo: bool = False,
    boot_config_fpath_during_hpo: Path = Path(),
    build_space_good_for_boot: bool = False,
    workload_timeouts: list[float] = [600.0],
    query_timeouts: list[float] = [30.0],
) -> dict[str, Any]:

    return {
        # Internal space versioning.
        "space_version": "2.0",
        "trace": True,
        "seed": seed,
        # For params that may differ between HPO, tune, and replay, I chose to represent them
        #   as dictionaries. I felt this was less confusing that overriding parts of the hpo_params
        #   during tune or replay. With the dictionary representation, we never override anything in
        #   hpo_params - we only ever add new fields to hpo_params.
        "enable_boot": {
            str(TuningMode.HPO): enable_boot_during_hpo,
        },
        "boot_config_fpath": {
            str(TuningMode.HPO): boot_config_fpath_during_hpo,
        },
        # Timeouts.
        "tune_duration": {
            str(TuningMode.HPO): tune_duration_during_hpo,
        },
        "workload_timeout": {
            str(TuningMode.HPO): tune.choice(workload_timeouts),
        },
        "query_timeout": tune.choice(query_timeouts),
        # Paths.
        "workload_path": str(workload_path),
        "pgconn_info": pgconn_info,
        "benchmark_config": benchmark_config,
        "benchbase_config": benchbase_config,
        # Embeddings.
        "embedder_path": tune.choice(map(str, embedder_path)),
        # Default quantization factor to use.
        "default_quantization_factor": 100,
        "system_knobs": sysknobs,
        # Horizon before resetting.
        "horizon": 5,
        # Workload Eval.
        "workload_eval_mode": tune.choice(["all", "all_enum"]),
        "workload_eval_inverse": tune.choice([False, True]),
        "workload_eval_reset": True,
        # Reward.
        "reward": tune.choice(["multiplier", "relative"]),
        "reward_scaler": tune.choice([1, 2, 10]),
        "workload_timeout_penalty": 1,
        "normalize_reward": tune.choice([False, True]),
        # State.
        "metric_state": tune.choice(
            ([] if build_space_good_for_boot else ["metric"])
            + ["structure", "structure_normalize"]
        ),
        "maximize_state": not benchmark_config.get("oltp_workload", False),
        # Whether to normalize state or not.
        "normalize_state": tune.sample_from(
            lambda spc: (
                False
                if spc["config"]["metric_state"] == "structure_normalize"
                else True
            )
        ),
        # LSC Parameters. The units for these are based on the embedding itself.
        # TODO(): Set these parameters based on the workload/embedding structure itself.
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
        "learning_rate": tune.choice([1e-3, 6e-4, 3e-5]),
        "critic_lr_scale": tune.choice([1.0, 2.5, 5.0]),
        "policy_l2_reg": tune.choice([0.01, 0.05]),
        # Discount.
        "gamma": tune.choice([0, 0.9, 0.95]),
        # Polyak averaging rate.
        "tau": tune.choice([0.995, 1.0]),
        # Replay Buffer Size.
        "buffer_size": 1_000_000,
        # Batch size.
        "batch_size": tune.choice([16, 32]),
        # Gradient Clipping.
        "grad_clip": tune.choice([1.0, 5.0, 10.0]),
        # Gradient steps per sample.
        "gradient_steps": tune.choice([1, 2, 4]),
        # Training steps.
        "train_freq_unit": tune.choice(["step", "episode"]),
        "train_freq_frequency": 1,
        # Target noise.
        "target_noise": {
            "target_noise_clip": tune.choice([0.05, 0.1, 0.15]),
            "target_policy_noise": tune.choice([0.15, 0.2]),
        },
        # Noise parameters.
        "noise_parameters": {
            "noise_type": tune.choice(["normal", "ou"]),
            "noise_sigma": tune.choice([0.05, 0.1, 0.15]),
        },
        "scale_noise_perturb": True,
        # Neighbor parameters.
        "neighbor_parameters": {
            "knob_num_nearest": tune.choice([10, 100]),
            "knob_span": tune.choice([1, 3]),
            "index_num_samples": 1,
            # Use index rules whenever we aren't optimizing OLTP.
            "index_rules": not benchmark_config.get("oltp_workload", False),
        },
        # Networks.
        "weight_init": tune.choice(["xavier_normal", "xavier_uniform", "orthogonal"]),
        "bias_zero": tune.choice([False, True]),
        "policy_weight_adjustment": tune.choice([1, 100]),
        "activation_fn": tune.choice(["gelu", "mish"]),
        "pi_arch": tune.choice(["128,128", "256,256", "512,512"]),
        "qf_arch": tune.choice(["256", "512", "1024"]),
    }


class TuneTimeoutChecker(object):
    def __init__(self, tune_duration: float) -> None:
        self.limit = (tune_duration * 3600) > 0
        self.remain = int(tune_duration * 3600)
        self.running = False
        self.start = 0.0

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
    def __init__(
        self,
        dbgym_cfg: DBGymConfig,
        tuning_mode: TuningMode,
        ray_trial_id: Optional[str] = None,
    ) -> None:
        """
        We use this object for HPO, tune, and replay. It behaves *slightly* differently
        depending on what it's used for, which is why we have the tuning_mode param.
        """
        self.dbgym_cfg = dbgym_cfg
        self.tuning_mode = tuning_mode

        if self.tuning_mode == TuningMode.HPO:
            assert (
                ray_trial_id != None
            ), "If we're doing HPO, we will create multiple TuneTrial() objects. We thus need to differentiate them somehow."
        else:
            assert (
                ray_trial_id is None
            ), "If we're not doing HPO, we (currently) will create only one TuneTrial() object. For clarity, we set ray_trial_id to None since ray_trial_id should not be used in this case."
        self.ray_trial_id = ray_trial_id

    def setup(self, hpo_params: dict[str, Any]) -> None:
        # Attach mythril directory to the search path.
        sys.path.append(os.path.expanduser(self.dbgym_cfg.dbgym_repo_path))

        torch.set_default_dtype(torch.float32)  # type: ignore[no-untyped-call]
        seed = (
            hpo_params["seed"]
            if hpo_params["seed"] != -1
            else np.random.randint(np.iinfo(np.int32).max)
        )
        np.random.seed(seed)
        torch.manual_seed(seed)

        tune_duration = hpo_params["tune_duration"][str(self.tuning_mode)]

        self.timeout_checker = TuneTimeoutChecker(tune_duration)
        self.artifact_manager, self.target_reset, self.env, self.agent, self.signal = (
            build_trial(
                self.dbgym_cfg,
                self.tuning_mode,
                seed=seed,
                hpo_params=hpo_params,
                ray_trial_id=self.ray_trial_id,
            )
        )
        logging.getLogger(DBGYM_LOGGER_NAME).info("%s", hpo_params)
        logging.getLogger(DBGYM_LOGGER_NAME).info(f"Seed: {seed}")

        # Attach the timeout checker and loggers.
        self.agent.set_timeout_checker(self.timeout_checker)
        self.agent.set_artifact_manager(self.artifact_manager)

        self.env_init = False
        self.start_time = time.time()
        self.step_count = 0

    def step(self) -> dict[Any, Any]:
        self.step_count += 1
        # Only measure the actual tuning time.
        self.timeout_checker.resume()

        episode = self.agent._episode_num
        it = self.agent.num_timesteps
        logging.getLogger(DBGYM_LOGGER_NAME).info(
            f"Starting episode: {episode+1}, iteration: {it+1}"
        )

        if not self.env_init:
            _, infos = self.env.reset()
            baseline_reward, baseline_metric = (
                infos["baseline_reward"],
                infos["baseline_metric"],
            )
            metric_reward_message = f"Baseline Metric: {baseline_metric}. Baseline Reward: {baseline_reward}"
            logging.getLogger(DBGYM_LOGGER_NAME).info(metric_reward_message)
            self.artifact_manager.log_to_replay_info(metric_reward_message)
            self.env_init = True

            assert (
                self.ray_trial_id != None
                if self.tuning_mode == TuningMode.HPO
                else True
            ), "If we're doing HPO, we need to ensure that we're passing a non-None ray_trial_id to stash_results() to avoid conflicting folder names."
            self.artifact_manager.stash_results(
                infos, name_override="baseline", ray_trial_id=self.ray_trial_id
            )
        else:
            self.agent.learn(self.env, total_timesteps=1, tuning_mode=self.tuning_mode)

        self.timeout_checker.pause()
        self.artifact_manager.advance()

        # Step telemetry that we care about.
        data = {
            "AgentEpisode": episode,
            "AgentTimesteps": it,
            "TrialStep": self.step_count,
            "Best Metric": (
                self.target_reset.real_best_metric if self.target_reset else -1
            ),
            "Best Seen Metric": (
                self.target_reset.best_metric if self.target_reset else -1
            ),
            "HoursElapsed": (time.time() - self.start_time) / 3600.0,
        }

        # If we've timed out. Note that we've timed out.
        if self.timeout_checker():
            self.cleanup()
            data[ray.tune.result.DONE] = True

        return data

    def cleanup(self) -> None:
        self.artifact_manager.flush()
        self.env.close()  # type: ignore[no-untyped-call]
        if Path(self.signal).exists():
            os.remove(self.signal)


# I want to pass dbgym_cfg into TuneOpt without putting it inside `hpo_params`. This is because it's a pain to turn DBGymConfig
#   into a nice dictionary of strings, and nothing in DBGymConfig would be relevant to someone checking the configs later
# Using a function to create a class is Ray's recommended way of doing this (see
#   https://discuss.ray.io/t/using-static-variables-to-control-trainable-subclass-in-ray-tune/808/4)
# If you don't create the class with a function, it doesn't work due to how Ray serializes classes
global_dbgym_cfg: DBGymConfig


def create_tune_opt_class(dbgym_cfg_param: DBGymConfig) -> Type[Trainable]:
    global global_dbgym_cfg
    global_dbgym_cfg = dbgym_cfg_param

    class TuneOpt(Trainable):
        dbgym_cfg = global_dbgym_cfg

        def setup(self, hpo_params: dict[str, Any]) -> None:
            self.trial = TuneTrial(
                TuneOpt.dbgym_cfg, TuningMode.HPO, ray_trial_id=self.trial_id
            )
            self.trial.setup(hpo_params)

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


def _tune_hpo(dbgym_cfg: DBGymConfig, hpo_args: AgentHPOArgs) -> None:
    with open_and_save(dbgym_cfg, hpo_args.sysknobs_path) as f:
        sysknobs = yaml.safe_load(f)["system_knobs"]

    with open_and_save(dbgym_cfg, hpo_args.benchmark_config_path) as f:
        benchmark_config = yaml.safe_load(f)
        is_oltp = benchmark_config["protox"]["query_spec"]["oltp_workload"]
        benchmark = [k for k in benchmark_config.keys()][0]
        benchmark_config = benchmark_config[benchmark]
        benchmark_config["benchmark"] = benchmark

    # TODO(phw2): read the dir hpo_args.embedder_path and get a list of embeddings
    embedder_path = [hpo_args.embedder_path]
    # TODO(phw2): make workload and query timeout params lists instead of just ints
    workload_timeouts = [hpo_args.workload_timeout]
    query_timeouts = [hpo_args.query_timeout]

    assert not is_oltp
    benchbase_config: dict[str, Any] = {}
    # This is commented out because OLTP is currently not implemented.
    # benchbase_config = (
    #     {
    #         "oltp_config": {
    #             "oltp_num_terminals": hpo_args.oltp_num_terminals,
    #             "oltp_duration": hpo_args.oltp_duration,
    #             "oltp_sf": hpo_args.oltp_sf,
    #             "oltp_warmup": hpo_args.oltp_warmup,
    #         },
    #         "benchbase_path": hpo_args.benchbase_path,
    #         "benchbase_config_path": hpo_args.benchbase_config_path,
    #     }
    #     if is_oltp
    #     else {}
    # )

    space = build_space(
        sysknobs,
        benchmark_config,
        hpo_args.workload_path,
        embedder_path,
        pgconn_info={
            "pristine_dbdata_snapshot_path": hpo_args.pristine_dbdata_snapshot_path,
            "dbdata_parent_dpath": hpo_args.dbdata_parent_dpath,
            "pgbin_path": hpo_args.pgbin_path,
        },
        benchbase_config=benchbase_config,
        tune_duration_during_hpo=hpo_args.tune_duration_during_hpo,
        seed=hpo_args.seed,
        enable_boot_during_hpo=hpo_args.enable_boot_during_hpo,
        boot_config_fpath_during_hpo=hpo_args.boot_config_fpath_during_hpo,
        build_space_good_for_boot=hpo_args.build_space_good_for_boot,
        workload_timeouts=workload_timeouts,
        query_timeouts=query_timeouts,
    )

    restart_ray(dbgym_cfg.root_yaml["ray_gcs_port"])
    ray.init(
        address=f"localhost:{dbgym_cfg.root_yaml['ray_gcs_port']}", log_to_driver=False
    )

    # Scheduler.
    scheduler = FIFOScheduler()  # type: ignore[no-untyped-call]

    # Search.
    search = BasicVariantGenerator(max_concurrent=hpo_args.max_concurrent)

    mode = "max" if is_oltp else "min"
    tune_config = TuneConfig(
        scheduler=scheduler,
        search_alg=search,
        num_samples=hpo_args.num_samples,
        max_concurrent_trials=hpo_args.max_concurrent,
        chdir_to_trial_dir=True,
        metric=METRIC_NAME,
        mode=mode,
    )

    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_config = RunConfig(
        name=f"ProtoxHPO_{dtime}",
        failure_config=FailureConfig(max_failures=0, fail_fast=True),
        sync_config=SyncConfig(),
        verbose=2,
        log_to_file=True,
        storage_path=str(dbgym_cfg.cur_task_runs_path("hpo_ray_results", mkdir=True)),
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
                logging.getLogger(DBGYM_LOGGER_NAME).error(f"Trial {results[i]} FAILED")
        assert False, "Encountered exceptions!"

    # Save the best params.json.
    best_result = results.get_best_result(metric=METRIC_NAME, mode=mode)
    best_params_generated_fpath = Path(best_result.path) / "params.json"
    # Before saving, copy it into run_*/[codebase]/data/. This way, save_file() called on
    #   params.json will link directly to run_*/[codebase]/data/params.json instead of to
    #   run_*/[codebase]/hpo_ray_results/TuneOpt*/.
    best_params_copy_fpath = (
        dbgym_cfg.cur_task_runs_data_path(mkdir=True) / "params.json"
    )
    shutil.copy(best_params_generated_fpath, best_params_copy_fpath)
    link_result(
        dbgym_cfg,
        best_params_copy_fpath,
        custom_result_name=get_default_hpoed_agent_params_fname(
            hpo_args.benchmark_name, hpo_args.workload_name
        )
        + ".link",
    )
    # We also link from run_*/[codebase]/data/params.json to run_*/[codebase]/hpo_ray_results/TuneOpt*/**/params.json.
    #   This way, when _manually_ looking through run_*/, we can see which HPO trial was
    #   responsible for creating params.json.
    best_params_link_fpath = (
        dbgym_cfg.cur_task_runs_data_path(mkdir=True) / "params.json.link"
    )
    os.symlink(best_params_generated_fpath, best_params_link_fpath)
