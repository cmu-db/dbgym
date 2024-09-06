"""Abstract base classes for RL algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from misc.utils import TuningMode
from tune.protox.agent.agent_env import AgentEnv
from tune.protox.agent.noise import ActionNoise
from tune.protox.env.artifact_manager import ArtifactManager


class BaseAlgorithm(ABC):
    """
    The base of RL algorithms
    :param seed: Seed for the pseudo random generators
    """

    def __init__(self, seed: Optional[int] = None):
        self.num_timesteps = 0
        self._total_timesteps = 0
        self.seed = seed
        self.action_noise: Optional[ActionNoise] = None
        self._last_obs: Optional[NDArray[np.float32]] = None
        self._episode_num = 0
        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int
        # The logger object
        self._logger: Optional[ArtifactManager] = None
        self.timeout_checker = None

    def set_logger(self, logger: Optional[ArtifactManager]) -> None:
        """
        Setter for for logger object.

        .. warning::
        """
        self._logger = logger

    @property
    def logger(self) -> ArtifactManager:
        """Getter for the logger object."""
        assert self._logger is not None
        return self._logger

    def set_timeout_checker(self, timeout_checker: Any) -> None:
        self.timeout_checker = timeout_checker

    def _setup_learn(
        self,
        env: AgentEnv,
        total_timesteps: int,
    ) -> int:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :return: Total timesteps
        """
        if self.action_noise is not None:
            self.action_noise.reset()

        # Make sure training timesteps are ahead of the internal counter
        total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if self._last_obs is None:
            (
                self._last_obs,
                _,
            ) = env.reset()  # pytype: disable=annotation-type-mismatch

        return total_timesteps

    @abstractmethod
    def learn(
        self, env: AgentEnv, total_timesteps: int, tuning_mode: TuningMode
    ) -> None:
        """
        Return a trained model.

        :param total_timesteps: The total number of samples (env steps) to train on
        :return: the trained model
        """
