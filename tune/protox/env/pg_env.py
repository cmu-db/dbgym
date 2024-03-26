import copy
import json
import time
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import gymnasium as gym
import psycopg
from plumbum import local

from misc.utils import DBGymConfig
from tune.protox.env.logger import Logger, time_record
from tune.protox.env.space.holon_space import HolonSpace
from tune.protox.env.space.state.space import StateSpace
from tune.protox.env.types import (
    EnvInfoDict,
    HolonAction,
    HolonStateContainer,
    TargetResetConfig,
)
from tune.protox.env.util.pg_conn import PostgresConn
from tune.protox.env.util.reward import RewardUtility
from tune.protox.env.workload import Workload


class PostgresEnv(gym.Env[Any, Any]):
    def __init__(
        self,
        dbgym_cfg: DBGymConfig,
        observation_space: StateSpace,
        action_space: HolonSpace,
        workload: Workload,
        horizon: int,
        reward_utility: RewardUtility,
        pgconn: PostgresConn,
        pqt: int,
        benchbase_config: dict[str, Any],
        logger: Optional[Logger] = None,
        replay: bool = False,
    ):
        super().__init__()

        self.dbgym_cfg = dbgym_cfg
        self.replay = replay
        self.logger = logger
        self.action_space = action_space
        self.observation_space = observation_space
        self.workload = workload
        self.horizon = horizon
        self.reward_utility = reward_utility

        self.benchbase_config = benchbase_config
        self.pgconn = pgconn
        self.pqt = pqt

        self.current_state: Optional[Any] = None
        self.baseline_metric: Optional[float] = None
        self.state_container: Optional[HolonStateContainer] = None

    def _restore_last_snapshot(self) -> None:
        assert self.horizon > 1 and self.workload.oltp_workload
        assert self.pgconn.restore_checkpointed_snapshot()
        assert isinstance(self.action_space, HolonSpace)

        self.state_container = self.action_space.generate_state_container(
            self.state_container,
            None,
            self.pgconn.conn(),
            self.workload.queries,
        )

        if self.logger:
            self.logger.get_logger(__name__).debug(
                f"[Restored snapshot] {self.state_container}"
            )

    @time_record("reset")
    def reset(  # type: ignore
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> Tuple[Any, EnvInfoDict]:
        reset_start = time.time()
        if self.logger:
            self.logger.get_logger(__name__).info(
                "Resetting database system state to snapshot."
            )
        super().reset(seed=seed)

        target_config: Optional[TargetResetConfig] = None
        if options is not None:
            target_config = TargetResetConfig(
                {
                    "metric": options.get("metric", None),
                    "env_state": options.get("env_state", None),
                    "config": options.get("config", None),
                }
            )

        self.current_step = 0
        info = EnvInfoDict({})

        if target_config is not None:
            metric = target_config["metric"]
            env_state = target_config["env_state"]
            config = target_config["config"]

            if self.workload.oltp_workload and self.horizon == 1:
                # Restore a pristine snapshot of the world if OTLP and horizon = 1
                self.pgconn.restore_pristine_snapshot()
            else:
                # Instead of restoring a pristine snapshot, just reset the knobs.
                # This in effect "resets" the baseline knob settings.
                self.pgconn.start_with_changes(conf_changes=[])

            # Maneuver the state into the requested state/config.
            assert isinstance(self.action_space, HolonSpace)
            sc = self.action_space.generate_state_container(
                self.state_container,
                None,
                self.pgconn.conn(),
                self.workload.queries,
            )
            config_changes, sql_commands = self.action_space.generate_plan_from_config(
                config, sc
            )
            assert self.shift_state(config_changes, sql_commands)

            # Note that we do not actually update the baseline metric/reward used by the reward
            # utility. This is so the reward is not stochastic with respect to the starting state.
            # This also means the reward is deterministic w.r.t to improvement.
            if self.reward_utility is not None:
                assert self.baseline_metric
                self.reward_utility.set_relative_baseline(
                    self.baseline_metric, prev_result=metric
                )

            self.state_container = copy.deepcopy(config)
            self.current_state = env_state.copy()
            if self.logger:
                self.logger.get_logger(__name__).debug(
                    "[Finished] Reset to state (config): %s", config
                )

        else:
            # Restore a pristine snapshot of the world.
            self.pgconn.restore_pristine_snapshot()
            assert not self.replay

            # On the first time, run the benchmark to get the baseline.
            assert isinstance(self.observation_space, StateSpace)
            assert isinstance(self.action_space, HolonSpace)

            # Get the stock state container.
            sc = self.action_space.generate_state_container(
                None, None, self.pgconn.conn(), self.workload.queries
            )
            default_action = self.action_space.null_action(sc)

            success, metric, _, results, _, query_metric_data = self.workload.execute(
                pgconn=self.pgconn,
                reward_utility=self.reward_utility,
                obs_space=self.observation_space,
                action_space=self.action_space,
                actions=[default_action],
                actions_names=["GlobalDual"],
                benchbase_config=self.benchbase_config,
                pqt=self.pqt,
                update=False,
                first=True,
            )

            # Ensure that the first run succeeds.
            assert success
            # Get the state.
            self.state_container = self.action_space.generate_state_container(
                self.state_container,
                None,
                self.pgconn.conn(),
                self.workload.queries,
            )
            state = self.observation_space.construct_offline(
                self.pgconn.conn(), results, self.state_container
            )

            # Set the metric workload.
            self.workload.set_workload_timeout(metric)

            self.reward_utility.set_relative_baseline(metric, prev_result=metric)
            _, reward = self.reward_utility(
                metric=metric, update=False, did_error=False
            )
            self.current_state = state.copy()
            info = EnvInfoDict(
                {
                    "baseline_metric": metric,
                    "baseline_reward": reward,
                    "query_metric_data": query_metric_data,
                    "results": results,
                    "prior_state_container": None,
                    "prior_pgconf": None,
                    "action_json": None,
                }
            )
            self.baseline_metric = metric

        assert self.state_container
        info["state_container"] = copy.deepcopy(self.state_container)
        return self.current_state, info

    @time_record("step_before_execution")
    def step_before_execution(self, action: HolonAction) -> Tuple[bool, EnvInfoDict]:
        # Log the action in debug mode.
        if self.logger:
            self.logger.get_logger(__name__).debug(
                "Selected action: %s", self.action_space.to_jsonable([action])
            )

        # Get the prior state.
        prior_state = copy.deepcopy(self.state_container)
        # Save the old configuration file.
        old_conf_path = f"{self.pgconn.pgdata_path}/postgresql.auto.conf"
        conf_path = f"{self.pgconn.pgdata_path}/postgresql.auto.old"
        local["cp"][old_conf_path, conf_path].run()

        # Figure out what we have to change to get to the new configuration.
        assert isinstance(self.action_space, HolonSpace)
        assert prior_state
        config_changes, sql_commands = self.action_space.generate_action_plan(
            action, prior_state
        )
        # Attempt to maneuver to the new state.
        success = self.shift_state(config_changes, sql_commands)
        return success, EnvInfoDict(
            {
                "attempted_changes": (config_changes, sql_commands),
                "prior_state_container": prior_state,
                "prior_pgconf": conf_path,
            }
        )

    @time_record("step_execute")
    def step_execute(
        self,
        setup_success: bool,
        actions: list[Tuple[str, HolonAction]],
        info: EnvInfoDict,
    ) -> Tuple[bool, EnvInfoDict]:
        if setup_success:
            assert isinstance(self.observation_space, StateSpace)
            assert isinstance(self.action_space, HolonSpace)
            # Evaluate the benchmark.
            start_time = time.time()
            (
                success,
                metric,
                reward,
                results,
                q_timeout,
                query_metric_data,
            ) = self.workload.execute(
                pgconn=self.pgconn,
                reward_utility=self.reward_utility,
                obs_space=self.observation_space,
                action_space=self.action_space,
                benchbase_config=self.benchbase_config,
                pqt=self.pqt,
                actions=[a[1] for a in actions],
                actions_names=[a[0] for a in actions],
                update=True,
            )
        else:
            # Illegal configuration.
            if self.logger:
                self.logger.get_logger(__name__).info(
                    "Found illegal configuration: %s", info["attempted_changes"]
                )
            success = False
            # Since we reached an invalid area, just set the next state to be the current state.
            metric, reward = self.reward_utility(did_error=True)
            results, q_timeout, query_metric_data = None, True, None

        info.update(
            EnvInfoDict(
                {
                    "metric": metric,
                    "q_timeout": q_timeout,
                    "query_metric_data": query_metric_data,
                    "reward": reward,
                    "results": results,
                    "action_json": json.dumps(
                        self.action_space.to_jsonable([a[1] for a in actions])
                    ),
                }
            )
        )
        return success, info

    @time_record("step_post_execute")
    def step_post_execute(
        self,
        success: bool,
        action: HolonAction,
        info: EnvInfoDict,
        soft: bool = False,
    ) -> Tuple[Any, float, bool, bool, EnvInfoDict]:
        if self.workload.oltp_workload and self.horizon > 1:
            # If horizon = 1, then we're going to reset anyways. So easier to just untar the original archive.
            # Restore the crisp and clean snapshot.
            # If we've "failed" due to configuration, then we will boot up the last "bootable" version.
            self._restore_last_snapshot()

        if success:
            if not soft:
                if not self.workload.oltp_workload:
                    # Update the workload metric timeout if we've succeeded.
                    self.workload.set_workload_timeout(info["metric"])

            # Get the current view of the state container.
            assert isinstance(self.action_space, HolonSpace)
            self.state_container = self.action_space.generate_state_container(
                self.state_container,
                action,
                self.pgconn.conn(),
                self.workload.queries,
            )

            # Now. The state container should be accurate.
            assert isinstance(self.observation_space, StateSpace)
            next_state = self.observation_space.construct_offline(
                self.pgconn.conn(), info["results"], self.state_container
            )
        else:
            assert self.current_state
            next_state = self.current_state.copy()

        if not soft:
            self.current_step = self.current_step + 1
        self.current_state = next_state
        return (
            self.current_state,
            info["reward"],
            (self.current_step >= self.horizon),
            not success,
            info,
        )

    def step(  # type: ignore
        self, action: HolonAction
    ) -> Tuple[Any, float, bool, bool, EnvInfoDict]:
        assert not self.replay
        success, info = self.step_before_execution(action)
        success, info = self.step_execute(success, [("PerQuery", action)], info)
        return self.step_post_execute(success, action, info)

    @time_record("shift_state")
    def shift_state(
        self,
        config_changes: list[str],
        sql_commands: list[str],
        dump_page_cache: bool = False,
        ignore_error: bool = False,
    ) -> bool:
        def attempt_checkpoint(conn_str: str) -> None:
            # CHECKPOINT to prevent the DBMS from entering a super slow shutdown
            # if a shift_state has failed.
            try:
                with psycopg.connect(
                    conn_str, autocommit=True, prepare_threshold=None
                ) as conn:
                    conn.execute("CHECKPOINT")
            except psycopg.OperationalError as e:
                if self.logger:
                    self.logger.get_logger(__name__).debug(f"[attempt_checkpoint]: {e}")
                time.sleep(5)

        shift_start = time.time()
        # First enforce the SQL command changes.
        for i, sql in enumerate(sql_commands):
            if self.logger:
                self.logger.get_logger(__name__).info(
                    f"Executing {sql} [{i+1}/{len(sql_commands)}]"
                )

            ret, stderr = self.pgconn.psql(sql)
            if ret == -1:
                if stderr:
                    print(stderr, flush=True)
                    assert (
                        "index row requires" in stderr
                        or "canceling statement" in stderr
                        # We've killed the index operation.
                        or "operational" in stderr
                    )
                    attempt_checkpoint(self.pgconn.connection)
                return False

            assert ret == 0, print(stderr)

        # Now try and perform the configuration changes.
        return self.pgconn.start_with_changes(
            conf_changes=config_changes,
            dump_page_cache=dump_page_cache,
            save_checkpoint=self.workload.oltp_workload and self.horizon > 1,
        )

    def close(self) -> None:
        self.pgconn.shutdown_postgres()
        local["rm"]["-rf", self.pgconn.pgdata_path].run()
        local["rm"]["-rf", f"{self.pgconn.pgdata_path}.tgz"].run()
        local["rm"]["-rf", f"{self.pgconn.pgdata_path}.tgz.tmp"].run()
