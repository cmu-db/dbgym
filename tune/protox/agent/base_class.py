"""Abstract base classes for RL algorithms."""
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import numpy as np
import torch as th
from gymnasium import spaces

from tune.protox.agent.noise import ActionNoise
from tune.protox.agent.policies import BasePolicy
from tune.protox.agent.type_aliases import GymEnv, Schedule
from tune.protox.agent.utils import (
    check_for_correct_spaces,
    get_device,
    get_schedule_fn,
    update_learning_rate,
)
from tune.protox.agent.vec_env import VecEnv

SelfBaseAlgorithm = TypeVar("SelfBaseAlgorithm", bound="BaseAlgorithm")


class BaseAlgorithm(ABC):
    """
    The base of RL algorithms

    :param policy: The policy model to use (MlpPolicy, ...)
    :param env: The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param seed: Seed for the pseudo random generators
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    # Policy aliases (see _get_policy_from_name())
    policy_aliases: Dict[str, Type[BasePolicy]] = {}
    policy: BasePolicy

    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str, None],
        learning_rate: Union[float, Schedule],
        policy_kwargs: Optional[Dict[str, Any]] = None,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        seed: Optional[int] = None,
        supported_action_spaces: Optional[Tuple[spaces.Space, ...]] = None,
    ):
        if isinstance(policy, str):
            self.policy_class = self._get_policy_from_name(policy)
        else:
            self.policy_class = policy

        self.device = get_device(device)
        self.env = None  # type: Optional[GymEnv]
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space: spaces.Space
        self.action_space: spaces.Space
        self.n_envs: int
        self.num_timesteps = 0
        # Used for updating schedules
        self._total_timesteps = 0
        # Used for computing fps, it is updated at each call of learn()
        self._num_timesteps_at_start = 0
        self.seed = seed
        self.action_noise: Optional[ActionNoise] = None
        self.start_time = None
        self.learning_rate = learning_rate
        self.lr_schedule = None  # type: Optional[Schedule]
        self._last_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._last_episode_starts = None  # type: Optional[np.ndarray]
        self._episode_num = 0
        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1
        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int
        # The logger object
        self._logger = None

        assert env is not None
        assert isinstance(env, VecEnv)

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = env.num_envs
        self.env = env

        if supported_action_spaces is not None:
            assert isinstance(self.action_space, supported_action_spaces), (
                f"The algorithm only supports {supported_action_spaces} as action spaces "
                f"but {self.action_space} was provided"
            )

        if not support_multi_env and self.n_envs > 1:
            raise ValueError(
                "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
            )

        # Catch common mistake: using MlpPolicy
        if policy in ["MlpPolicy"] and isinstance(self.observation_space, spaces.Dict):
            raise ValueError(f"You must use `MultiInputPolicy` when working with dict observation space, not {policy}")

        if isinstance(self.action_space, spaces.Box):
            assert np.all(
                np.isfinite(np.array([self.action_space.low, self.action_space.high]))
            ), "Continuous action space must have a finite lower and upper bound"

    @abstractmethod
    def _setup_model(self) -> None:
        """Create networks, buffer and optimizers."""

    def set_logger(self, logger) -> None:
        """
        Setter for for logger object.

        .. warning::
        """
        self._logger = logger

    @property
    def logger(self):
        """Getter for the logger object."""
        assert self._logger is not None
        return self._logger

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))

    def _get_policy_from_name(self, policy_name: str) -> Type[BasePolicy]:
        """
        Get a policy class from its name representation.

        The goal here is to standardize policy naming, e.g.
        all algorithms can call upon "MlpPolicy"
        and they receive respective policies that work for them.

        :param policy_name: Alias of the policy
        :return: A policy class (type)
        """

        if policy_name in self.policy_aliases:
            return self.policy_aliases[policy_name]
        else:
            raise ValueError(f"Policy {policy_name} unknown")

    def _setup_learn(
        self,
        total_timesteps: int,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> int:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps
        """
        self.start_time = time.time_ns()

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs, _ = self.env.reset()  # pytype: disable=annotation-type-mismatch
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)

        return total_timesteps

    def get_env(self) -> Optional[VecEnv]:
        """
        Returns the current environment (can be None if not defined).

        :return: The current environment
        """
        return self.env

    def set_env(self, env: GymEnv, force_reset: bool = True) -> None:
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.
        Furthermore wrap any non vectorized env into a vectorized
        checked parameters:
        - observation_space
        - action_space

        :param env: The environment for learning a policy
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        """
        # if it is not a VecEnv, make it a VecEnv
        # and do other transformations (dict obs)  if needed
        assert isinstance(env, VecEnv)
        assert env.num_envs == self.n_envs, (
            "The number of environments to be set is different from the number of environments in the model: "
            f"({env.num_envs} != {self.n_envs}), whereas `set_env` requires them to be the same. To load a model with "
            f"a different number of environments, you must use `{self.__class__.__name__}.load(path, env)` instead"
        )
        # Check that the observation spaces match
        check_for_correct_spaces(env, self.observation_space, self.action_space)

        # Discard `_last_obs`, this will force the env to reset before training
        # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        if force_reset:
            self._last_obs = None

        self.n_envs = env.num_envs
        self.env = env

    @abstractmethod
    def learn(
        self: SelfBaseAlgorithm,
        total_timesteps: int,
        log_interval: int = 100,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfBaseAlgorithm:
        """
        Return a trained model.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param log_interval: The number of episodes before logging.
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: the trained model
        """
