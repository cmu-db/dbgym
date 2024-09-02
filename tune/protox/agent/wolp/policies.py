import time
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from numpy.typing import NDArray
from torch.optim import Optimizer  # type: ignore[attr-defined]

from tune.protox.agent.buffers import ReplayBufferSamples
from tune.protox.agent.noise import ActionNoise
from tune.protox.agent.policies import Actor, BaseModel, ContinuousCritic
from tune.protox.agent.utils import polyak_update
from tune.protox.env.logger import Logger, time_record
from tune.protox.env.space.holon_space import HolonSpace
from tune.protox.env.types import (
    DEFAULT_NEIGHBOR_PARAMETERS,
    HolonAction,
    NeighborParameters,
)

DETERMINISTIC_NEIGHBOR_PARAMETERS = {
    "knob_num_nearest": 1,
    "knob_span": 0,
    "index_num_samples": 1,
    "index_rules": False,
}


class WolpPolicy(BaseModel):
    """
    Policy class (with both actor and critic) for Wolp.

    :param observation_space: Observation space
    :param action_space: Action space
    :param actor
    :param actor_target
    :param critic
    :param critic_target
    """

    def __init__(
        self,
        observation_space: spaces.Space[Any],
        action_space: spaces.Space[Any],
        actor: Actor,
        actor_target: Actor,
        actor_optimizer: Optimizer,
        critic: ContinuousCritic,
        critic_target: ContinuousCritic,
        critic_optimizer: Optimizer,
        grad_clip: float = 1.0,
        policy_l2_reg: float = 0.0,
        tau: float = 0.005,
        gamma: float = 0.99,
        logger: Optional[Logger] = None,
    ):
        super().__init__(observation_space, action_space)
        self.actor = actor
        self.actor_target = actor_target
        self.actor_optimizer = actor_optimizer
        self.critic = critic
        self.critic_target = critic_target
        self.critic_optimizer = critic_optimizer
        self.logger = logger

        self.grad_clip = grad_clip
        self.policy_l2_reg = policy_l2_reg
        self.tau = tau
        self.gamma = gamma

        # Log all the networks.
        if self.logger:
            self.logger.get_logger(__name__).info("Actor: %s", self.actor)
            self.logger.get_logger(__name__).info("Critic: %s", self.critic)

    def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        raise NotImplementedError()

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode

    @time_record("discriminate")
    def discriminate(
        self,
        use_target: bool,
        states: th.Tensor,
        embed_actions: th.Tensor,
        actions_dim: th.Tensor,
        env_actions: list[HolonAction],
    ) -> tuple[list[HolonAction], th.Tensor]:
        states_tile = states.repeat_interleave(actions_dim, dim=0)
        if use_target:
            next_q_values = th.cat(
                self.critic_target(states_tile, embed_actions), dim=1
            )
            assert not th.isnan(next_q_values).any()
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
        else:
            next_q_values = th.cat(self.critic(states_tile, embed_actions), dim=1)
            assert not th.isnan(next_q_values).any()
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

        env_splitter = [0] + list(actions_dim.cumsum(dim=0))
        if env_actions is not None:
            split_env_actions = [
                env_actions[start:end]
                for start, end in zip(env_splitter[:-1], env_splitter[1:])
            ]
        # Split the actions.
        split_embed_actions = th.split(embed_actions, actions_dim.tolist())
        # Find the maximizing q-value action.
        actions_eval_split = th.split(next_q_values, actions_dim.tolist())
        max_indices = [th.argmax(split) for split in actions_eval_split]
        # Find the maximal action.
        if env_actions is not None:
            env_actions = [
                split_env_actions[i][max_indices[i]] for i in range(len(max_indices))
            ]
        embed_actions = th.stack(
            [split_embed_actions[i][max_indices[i]] for i in range(len(max_indices))]
        )
        assert states.shape[0] == embed_actions.shape[0]
        return env_actions, embed_actions

    def wolp_act(
        self,
        states: Union[th.Tensor, NDArray[np.float32]],
        use_target: bool = False,
        action_noise: Optional[Union[ActionNoise, th.Tensor]] = None,
        neighbor_parameters: NeighborParameters = DEFAULT_NEIGHBOR_PARAMETERS,
        random_act: bool = False,
    ) -> tuple[list[HolonAction], th.Tensor]:
        # Get the tensor representation.
        start_time = time.time()
        if not isinstance(states, th.Tensor):
            thstates = self.obs_to_tensor(states)
        else:
            thstates = states

        if random_act:
            assert hasattr(self.action_space, "sample_latent")
            raw_action = self.action_space.sample_latent()
            raw_action = th.as_tensor(raw_action).float()
        elif use_target:
            raw_action = self.actor_target(thstates)
        else:
            raw_action = self.actor(thstates)

        # Transform and apply the noise.
        noise = (
            None
            if action_noise is None
            else (
                action_noise
                if isinstance(action_noise, th.Tensor)
                else th.as_tensor(action_noise())
            )
        )
        if noise is not None and len(noise.shape) == 1:
            # Insert a dimension.
            noise = noise.view(-1, *noise.shape)

        if noise is not None and self.logger is not None:
            self.logger.get_logger(__name__).debug(
                f"Perturbing with noise class {action_noise}"
            )

        assert hasattr(self.action_space, "transform_noise")
        raw_action = self.action_space.transform_noise(raw_action, noise=noise)

        # Smear the action.
        if TYPE_CHECKING:
            assert isinstance(self.action_space, HolonSpace)
        env_actions, sample_actions, actions_dim = self.action_space.neighborhood(
            raw_action, neighbor_parameters
        )

        if self.logger is not None:
            # Log the neighborhood we are observing.
            self.logger.get_logger(__name__).debug(f"Neighborhood Sizes {actions_dim}")

        if random_act:
            # If we want a random action, don't use Q-value estimate.
            rand_act = np.random.randint(0, high=len(env_actions))
            return [env_actions[rand_act]], sample_actions[rand_act : rand_act + 1]

        assert thstates.shape[0] == actions_dim.shape[0]
        assert len(thstates.shape) == 2
        env_actions, embed_actions = self.discriminate(
            use_target, thstates, sample_actions, actions_dim, env_actions
        )
        assert not np.isnan(embed_actions).any()
        return env_actions, embed_actions

    @time_record("train_critic")
    def train_critic(
        self,
        replay_data: ReplayBufferSamples,
        neighbor_parameters: NeighborParameters,
        target_action_noise: Optional[ActionNoise] = None,
    ) -> Any:
        with th.no_grad():
            # wolp_act() actually gives both the env and the embedding actions.
            # We evaluate the critic on the embedding action and not the environment action.
            _, embed_actions = self.wolp_act(
                replay_data.next_observations,
                use_target=True,
                action_noise=target_action_noise,
                neighbor_parameters=neighbor_parameters,
            )

            # Compute the next Q-values: min over all critics targets
            next_q_values = th.cat(
                self.critic_target(replay_data.next_observations, embed_actions), dim=1
            )
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            target_q_values = (
                replay_data.rewards
                + (1 - replay_data.dones) * self.gamma * next_q_values
            )

        embeds = replay_data.actions.float()

        # Get current Q-values estimates for each critic network
        current_q_values = self.critic(replay_data.observations, embeds)
        # Compute critic loss.
        critic_losses = [
            F.mse_loss(current_q, target_q_values) for current_q in current_q_values
        ]
        critic_loss = cast(th.Tensor, sum(critic_losses))

        # Optimize the critics
        self.critic_optimizer.zero_grad()
        assert not th.isnan(critic_loss).any()
        critic_loss.backward()  # type: ignore
        th.nn.utils.clip_grad_norm_(list(self.critic.parameters()), self.grad_clip, error_if_nonfinite=True)
        self.critic.check_grad()
        self.critic_optimizer.step()
        return critic_loss

    @time_record("train_actor")
    def train_actor(self, replay_data: ReplayBufferSamples) -> Any:
        # Get the current action representation.
        embeds = replay_data.actions.float()

        # Compute actor loss
        raw_actions = self.actor(replay_data.observations)

        if "lsc" in replay_data.infos[0]:
            lscs = (
                th.as_tensor(np.array([i["lsc"] for i in replay_data.infos]))
                .float()
                .view(-1, 1)
            )
            # TODO(wz2,PROTOX_DELTA): Assume that we're looking at the "center".
            # Practically, maybe we should check through the critic what would actually be selected here.
            # The paper uses the historic action. This refactor assumes the neighborhood center.
            assert hasattr(self.action_space, "pad_center_latent")
            raw_actions = self.action_space.pad_center_latent(raw_actions, lscs)

        actor_loss = -self.critic.q1_forward(
            replay_data.observations, raw_actions
        ).mean()

        # Attach l2.
        if self.policy_l2_reg > 0:
            for param in self.actor.parameters():
                actor_loss += 0.5 * (param**2).sum()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        assert not th.isnan(actor_loss).any()
        actor_loss.backward()  # type: ignore
        th.nn.utils.clip_grad_norm_(list(self.actor.parameters()), self.grad_clip, error_if_nonfinite=True)
        self.actor.check_grad()
        self.actor_optimizer.step()
        return actor_loss

    def polyak_update(self) -> None:
        polyak_update(
            self.critic.parameters(), self.critic_target.parameters(), self.tau
        )
        polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
