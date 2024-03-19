import numpy as np
import json
import copy
from typing import Any, Optional, Tuple, Union

import gymnasium as gym
import torch

from tune.protox.env.logger import Logger
from tune.protox.env.pg_env import PostgresEnv
from tune.protox.env.space.holon_space import HolonSpace
from tune.protox.env.space.primitive import SettingType, is_binary_enum, is_knob_enum
from tune.protox.env.space.primitive.knob import CategoricalKnob, Knob
from tune.protox.env.space.state.space import StateSpace
from tune.protox.env.space.utils import parse_access_methods
from tune.protox.env.types import (
    HolonAction,
    BestQueryRun,
    QuerySpaceKnobAction,
    KnobSpaceAction,
    KnobSpaceContainer,
    QueryTableAccessMap,
    EnvInfoDict,
)


def _mutilate_action_with_metrics(
    action_space: HolonSpace,
    action: HolonAction,
    query_metric_data: Optional[dict[str, BestQueryRun]],
    timeout_qknobs: Optional[QuerySpaceKnobAction] = None,
) -> HolonAction:

    if query_metric_data is not None:
        extract_q_knobs = action_space.extract_query(action)
        assert extract_q_knobs

        processed = set()
        for q, data in query_metric_data.items():
            if not data.timeout:
                assert data.query_run
                pqk = data.query_run.qknobs
                for k, v in data.query_run.qknobs.items():
                    # Implant the best.
                    extract_q_knobs[k] = v
            processed.add(q)

        if timeout_qknobs:
            qspace = action_space.get_query_space()
            assert timeout_qknobs

            all_qids = set([k.query_name for k in timeout_qknobs.keys()]) - processed
            for qid in all_qids:
                qid_rest = {
                    k: v
                    for k, v in timeout_qknobs.items()
                    if k.name().startswith(f"{qid}_")
                }
                for k, v in qid_rest.items():
                    # Implant the reset target for queries we didn't see.
                    extract_q_knobs[k] = v

        action = action_space.replace_query(action, extract_q_knobs)
    return action


def _regress_query_knobs(
    qknobs: QuerySpaceKnobAction,
    sysknobs: Union[KnobSpaceAction, KnobSpaceContainer],
    ams: QueryTableAccessMap,
    logger: Optional[Logger] = None,
) -> QuerySpaceKnobAction:
    global_qknobs = {}
    for knob, _ in qknobs.items():
        if knob.knob_name in sysknobs:
            # Defer to system using the base knob (without table/query prefix).
            global_qknobs[knob] = sysknobs[knob.knob_name]
        elif knob.knob_type in [
            SettingType.SCANMETHOD_ENUM,
            SettingType.SCANMETHOD_ENUM_CATEGORICAL,
        ]:
            assert "_scanmethod" in knob.knob_name
            alias = knob.knob_name.split("_scanmethod")[0]
            qid_prefix = knob.query_name
            assert qid_prefix

            if alias in ams[qid_prefix]:
                value = 1.0 if "Index" in ams[qid_prefix][alias] else 0.0
            else:
                # Log out the missing alias for debugging reference.
                if logger:
                    logger.get_logger(__name__).debug(
                        f"Found missing {alias} in the parsed {ams}."
                    )
                value = 0.0
            global_qknobs[knob] = value
        elif knob.knob_type == SettingType.BOOLEAN:
            global_qknobs[knob] = 1.0
        elif knob.knob_name == "random_page_cost":
            global_qknobs[knob] = knob.project_scraped_setting(4.0)
        elif knob.knob_name == "seq_page_cost":
            global_qknobs[knob] = knob.project_scraped_setting(1.0)
        elif knob.knob_name == "hash_mem_multiplier":
            global_qknobs[knob] = knob.project_scraped_setting(2.0)
        elif isinstance(knob, CategoricalKnob):
            global_qknobs[knob] = knob.default_value
    assert len(global_qknobs) == len(qknobs)
    return QuerySpaceKnobAction(global_qknobs)


class MQOWrapper(gym.Wrapper[Any, Any, Any, Any]):
    def __init__(
        self,
        workload_eval_mode: str,
        workload_eval_inverse: bool,
        workload_eval_reset: bool,
        pqt: int,
        benchbase_config: dict[str, Any],
        env: gym.Env[Any, Any],
        logger: Optional[Logger],
    ):
        assert isinstance(env, PostgresEnv) or isinstance(env.unwrapped, PostgresEnv), print(
            "MQOPostgresEnv must be directly above PostgresEnv"
        )
        super().__init__(env)

        self.workload_eval_mode = workload_eval_mode
        assert self.workload_eval_mode in [
            "all",
            "all_enum",
            "global_dual",
            "prev_dual",
            "pq",
        ]

        self.workload_eval_mode = workload_eval_mode
        self.workload_eval_inverse = workload_eval_inverse
        self.workload_eval_reset = workload_eval_reset
        self.pqt = pqt
        self.benchbase_config = benchbase_config
        self.best_observed: dict[str, BestQueryRun] = {}
        self.logger = logger

    def _update_best_observed(self, query_metric_data: dict[str, BestQueryRun]) -> None:
        if query_metric_data is not None:
            for q, data in query_metric_data.items():
                if q not in self.best_observed:
                    self.best_observed[q] = BestQueryRun(data.query_run, data.runtime, data.timeout, None, None)
                elif not data.timeout:
                    qobs = self.best_observed[q]
                    assert qobs.runtime and data.runtime
                    if data.runtime < qobs.runtime:
                        self.best_observed[q] = BestQueryRun(data.query_run, data.runtime, data.timeout, None, None)

    def step( # type: ignore
        self,
        action: HolonAction,
    ) -> Tuple[Any, float, bool, bool, EnvInfoDict]:
        # Step based on the "global" action.
        assert isinstance(self.unwrapped, PostgresEnv)
        success, info = self.unwrapped.step_before_execution(action)
        prior_state = info["prior_state_container"]
        timeout_qknobs = None

        assert isinstance(self.action_space, HolonSpace)
        extract_q_knobs = self.action_space.extract_query(action)
        extract_knobs = self.action_space.extract_knobs(action)
        assert extract_q_knobs
        assert extract_knobs

        runs = []
        if prior_state and self.workload_eval_mode in ["all", "all_enum", "prev_dual"]:
            # Load the prior knobs.
            runs.append(
                (
                    "PrevDual",
                    self.action_space.replace_query(
                        action, self.action_space.extract_query(prior_state)
                    ),
                )
            )

            # FIXME(wz2): Default, restore towards what we've learned from the last step.
            # We can optionally also restore towards the "default" optimizer behavior.
            timeout_qknobs = self.action_space.extract_query(prior_state)

        if self.workload_eval_mode in ["all", "all_enum", "global_dual"]:
            # Load the global (optimizer) knobs.
            qid_ams = parse_access_methods(
                self.unwrapped.pgconn.conn(), self.unwrapped.workload.queries
            )
            runs.append(
                (
                    "GlobalDual",
                    self.action_space.replace_query(
                        action,
                        _regress_query_knobs(
                            extract_q_knobs, extract_knobs, qid_ams, self.logger
                        ),
                    ),
                )
            )

        # The requested agent.
        runs.append(("PerQuery", copy.deepcopy(action)))

        if self.workload_eval_inverse:
            # The selected inverse.
            runs.append(
                (
                    "PerQueryInverse",
                    self.action_space.replace_query(
                        action, QuerySpaceKnobAction({k: k.invert(v) for k, v in extract_q_knobs.items()})
                    ),
                )
            )

        if self.workload_eval_mode in ["all_enum"]:
            # Construct top-enum and bottom-enum knobs.
            def transmute(
                k: Union[Knob, CategoricalKnob], v: Any, top: bool = True
            ) -> Any:
                if not is_knob_enum(k.knob_type):
                    return v
                elif is_binary_enum(k.knob_type) and top:
                    return 1
                elif is_binary_enum(k.knob_type) and not top:
                    return 0
                else:
                    assert isinstance(k, CategoricalKnob)
                    return k.sample()

            runs.append(
                (
                    "TopEnum",
                    self.action_space.replace_query(
                        action,
                        QuerySpaceKnobAction({
                            k: transmute(k, v, top=True)
                            for k, v in extract_q_knobs.items()
                        }),
                    ),
                )
            )

            runs.append(
                (
                    "BottomEnum",
                    self.action_space.replace_query(
                        action,
                        QuerySpaceKnobAction({
                            k: transmute(k, v, top=False)
                            for k, v in extract_q_knobs.items()
                        }),
                    ),
                )
            )

        # Execute.
        success, info = self.unwrapped.step_execute(success, runs, info)
        if info["query_metric_data"]:
            self._update_best_observed(info["query_metric_data"])

        action = _mutilate_action_with_metrics(
            self.action_space, action, info["query_metric_data"], timeout_qknobs
        )

        with torch.no_grad():
            # Pass the mutilated action back through.
            assert isinstance(self.action_space, HolonSpace)
            info["action_json"] = json.dumps(self.action_space.to_jsonable([action]))
            info["maximal_embed"] = self.action_space.to_latent([action])

        return self.unwrapped.step_post_execute(success, action, info)

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[Any, EnvInfoDict]: # type: ignore
        assert isinstance(self.unwrapped, PostgresEnv)
        # First have to shift to the new state.
        state, info = self.unwrapped.reset(*args, **kwargs)

        # Now we conditionally evaluate.
        if self.workload_eval_reset and (kwargs and ("options" in kwargs) and (kwargs["options"]) and ("query_metric_data" in kwargs["options"])):
            assert isinstance(self.action_space, HolonSpace)
            assert isinstance(self.observation_space, StateSpace)

            # Get a null action.
            null_action = self.action_space.null_action(info["state_container"])
            # Reset-to
            reset_qknob = self.action_space.extract_query(info["state_container"])
            assert reset_qknob

            best_qknobs = self.action_space.extract_query(null_action)
            assert best_qknobs
            for qid, qr in self.best_observed.items():
                assert qr.query_run
                best_qknobs.update(qr.query_run.qknobs)

            # Replace into the null action.
            runs = [
                (
                    "ResetPerQuery",
                    self.action_space.replace_query(null_action, best_qknobs),
                )
            ]

            # This isn't ideal, but directly invoke execute() on the underlying workload object.
            # This will tell us whether the best_observed is better than what
            # we are actually resetting towards or not in target_metric_data.
            (
                success,
                metric,
                _,
                results,
                _,
                target_metric_data,
            ) = self.unwrapped.workload.execute(
                pgconn=self.unwrapped.pgconn,
                reward_utility=self.unwrapped.reward_utility,
                obs_space=self.observation_space,
                action_space=self.action_space,
                actions=[r[1] for r in runs],
                actions_names=[r[0] for r in runs],
                benchbase_config=self.benchbase_config,
                pqt=self.pqt,
                reset_metrics=kwargs["options"]["query_metric_data"],
                update=False,
                first=False,
            )

            # reset_qknob -- is the per query from the configuration being reset.
            # runs[0][1]/best_qknobs -- is the best observed.
            #
            # Merge the optimality between best_observed and what we are resetting
            # to based on the feedback in target_metric_data.
            action = _mutilate_action_with_metrics(
                self.action_space, runs[0][1], target_metric_data, reset_qknob
            )

            # Reward should be irrelevant. If we do accidentally use it, cause an error.
            info = EnvInfoDict({"metric": metric, "reward": np.inf, "results": results})
            # Use this to adjust the container and state but don't shift the step.
            state, _, _, _, info = self.unwrapped.step_post_execute(
                True, action, info, soft=True
            )

            if self.logger:
                self.logger.get_logger(__name__).debug("Maximized on reset.")

        return state, info
