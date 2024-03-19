from typing import Any, Dict, List, Optional, NamedTuple, NDArray, cast
import copy

import numpy as np
import torch as th

from tune.protox.agent.type_aliases import ReplayBufferSamples

class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    infos: List[dict[str, Any]]


class ReplayBuffer:
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: list[int],
        action_dim: int = 0,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape

        assert action_dim > 0
        self.action_dim = action_dim
        self.pos = 0
        self.full = False

        # Adjust buffer size
        self.buffer_size = buffer_size

        self.observations = np.zeros(
            (self.buffer_size, *self.obs_shape), dtype=np.float32
        )

        self.next_observations = np.zeros(
            (self.buffer_size, *self.obs_shape), dtype=np.float32
        )
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)

        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size), dtype=np.float32)
        self.infos: list[Optional[dict[str, Any]]] = [None] * self.buffer_size

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(
        self,
        obs: NDArray[np.float32],
        next_obs: NDArray[np.float32],
        action: NDArray[np.float32],
        reward: float,
        done: bool,
        infos: Dict[str, Any],
    ) -> None:
        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.infos[self.pos] = copy.deepcopy(infos)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: NDArray[np.int32]) -> ReplayBufferSamples:
        next_obs = self.next_observations[batch_inds, :]

        data = (
            self.observations[batch_inds, :],
            self.actions[batch_inds, :],
            next_obs,
            self.dones[batch_inds].reshape(-1, 1),
            self.rewards[batch_inds].reshape(-1, 1),
            cast(list[dict[str, Any]], [self.infos[x] for x in batch_inds]),
        )
        return ReplayBufferSamples(
            observations=self.to_torch(data[0]),
            actions=self.to_torch(data[1]),
            next_observations=self.to_torch(data[2]),
            dones=self.to_torch(data[3]),
            rewards=self.to_torch(data[4]),
            infos=data[-1],
        )

    def to_torch(self, array: NDArray[np.float32]) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :return:
        """
        if copy:
            return th.tensor(array)
        return th.as_tensor(array)
