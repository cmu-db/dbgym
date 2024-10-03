import logging
from typing import Any, Optional, Tuple

import gymnasium as gym

from tune.protox.env.artifact_manager import ArtifactManager
from tune.protox.env.lsc.lsc import LSC
from tune.protox.env.target_reset.target_reset_wrapper import TargetResetWrapper


class LSCWrapper(gym.Wrapper[Any, Any, Any, Any]):
    def __init__(self, lsc: LSC, env: gym.Env[Any, Any], artifact_manager: Optional[ArtifactManager]):
        assert not isinstance(env, TargetResetWrapper)
        super().__init__(env)
        self.lsc = lsc
        self.artifact_manager = artifact_manager

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        state, info = self.env.reset(*args, **kwargs)
        self.lsc.reset()

        state["lsc"] = self.lsc.current_scale()
        lsc = state["lsc"]
        logging.debug(f"Attaching LSC: {lsc}")

        return state, info

    def step(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        state, reward, term, trunc, info = self.env.step(*args, **kwargs)

        # Remember the LSC when we considered this action.
        info["lsc"] = self.lsc.current_scale()
        old_bias = self.lsc.current_bias()
        old_lsc = info["lsc"]

        # Store the new LSC.
        self.lsc.advance()
        state["lsc"] = self.lsc.current_scale()
        new_bias = self.lsc.current_bias()

        lsc = state["lsc"]
        logging.debug(
            f"Shifting LSC: {old_lsc} ({old_bias}) -> {lsc} ({new_bias})"
        )

        return state, float(reward), term, trunc, info
