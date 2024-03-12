import logging

import gymnasium as gym
from tune.protox.agent.env_util import unwrap_to_base_env
import random


class TargetStateResetWrapper(gym.core.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        maximize_state: False,
        reward_utility: None,
        maximize_knobs_only: False,
        start_reset: False,
    ):
        gym.Wrapper.__init__(self, env)
        self.maximize_state = maximize_state
        self.maximize_knobs_only = maximize_knobs_only
        self.start_reset = start_reset
        self.reward_utility = reward_utility
        self.tracked_states = None
        self.best_metric = None
        self.real_best_metric = None
        self.pg_env = unwrap_to_base_env(env)

    def save_state(self):
        return {
            "tracked_states": self.tracked_states,
            "best_metric": self.best_metric,
            "real_best_metric": self.real_best_metric,
        }

    def load_state(self, d):
        self.tracked_states = d["tracked_states"]
        self.best_metric = d["best_metric"]
        self.real_best_metric = d["real_best_metric"]

    def _get_state(self):
        state = self.pg_env.env_spec.action_space.get_state(self.pg_env)
        return state

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        accum_metric = infos.get("accum_metric", None)
        assert self.best_metric is not None
        q_timeout = infos.get("q_timeout", False)

        metric = infos["metric"]
        if self.reward_utility.is_perf_better(metric, self.best_metric):
            self.best_metric = infos["metric"]
            if not q_timeout:
                self.real_best_metric = self.best_metric

            if self.maximize_state:
                logging.info(f"Found new maximal state with {metric}.")
                assert self.tracked_states is not None
                state = self._get_state()
                if self.maximize_knobs_only:
                    assert self.pg_env.env_spec.action_space.get_knob_space() is not None
                    state = tuple([state[0]] + [[]]*int(len(state)-1))
                if self.start_reset:
                    self.tracked_states = [self.tracked_states[0], (metric, obs, accum_metric, state)]
                else:
                    self.tracked_states = [(metric, obs, accum_metric, state)]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        if self.tracked_states is None:
            # First time.
            state, info = self.env.reset(**kwargs)
            self.best_metric = info["baseline_metric"]
            self.real_best_metric = self.best_metric

            self.tracked_states = [
                (self.best_metric, state.copy(), info.get("accum_metric", None), self._get_state())
            ]
        else:
            metric, state, accum_metric, config = random.choice(self.tracked_states)
            if kwargs is None or "options" not in kwargs or kwargs["options"] is None:
                kwargs = {}
                kwargs["options"] = {}

            kwargs["options"]["metric"] = metric
            kwargs["options"]["state"] = state
            kwargs["options"]["config"] = config
            kwargs["options"]["accum_metric"] = accum_metric
            state, info = self.env.reset(**kwargs)
        return state, info
