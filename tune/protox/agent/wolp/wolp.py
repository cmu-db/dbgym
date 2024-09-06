from copy import deepcopy
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch as th
from numpy.typing import NDArray

from tune.protox.agent.agent_env import AgentEnv
from tune.protox.agent.buffers import ReplayBuffer
from tune.protox.agent.noise import ActionNoise
from tune.protox.agent.off_policy_algorithm import OffPolicyAlgorithm
from tune.protox.agent.wolp.policies import (
    DETERMINISTIC_NEIGHBOR_PARAMETERS,
    WolpPolicy,
)


class Wolp(OffPolicyAlgorithm):
    """
    Wolpertinger DDPG based on Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: The policy model
    :param replay_buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param seed: Seed for the pseudo random generators
    """

    def __init__(
        self,
        policy: WolpPolicy,
        replay_buffer: ReplayBuffer,
        learning_starts: int = 100,
        batch_size: int = 100,
        train_freq: tuple[int, str] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        target_action_noise: Optional[ActionNoise] = None,
        seed: Optional[int] = None,
        neighbor_parameters: dict[str, Any] = {},
        ray_trial_id: Optional[str] = None,
    ):
        super().__init__(
            policy,
            replay_buffer,
            learning_starts,
            batch_size,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            seed=seed,
            ray_trial_id=ray_trial_id,
        )

        self.target_action_noise = target_action_noise
        self.neighbor_parameters = neighbor_parameters

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: NDArray[np.float32],
        new_obs: NDArray[np.float32],
        reward: float,
        dones: bool,
        infos: dict[str, Any],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because AgentEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Avoid changing the original ones
        self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward
        assert self._last_original_obs is not None

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the Env resets automatically, new_obs is already the
        # first observation of the next episode
        if dones:
            assert infos.get("terminal_observation") is not None
            assert not isinstance(next_obs, dict)
            next_obs = infos["terminal_observation"]

        if "maximal_embed" in infos and infos["maximal_embed"] is not None:
            buffer_action = infos["maximal_embed"]

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts:
            # Warmup phase
            self.policy.set_training_mode(False)
            with th.no_grad():
                # Not sure how good of an idea it is to inject more stochasticity
                # into the randomness of an action. Just let the star map guide you.
                env_action, embed_action = self.policy.wolp_act(
                    self._last_obs,
                    use_target=False,
                    action_noise=None,
                    neighbor_parameters=DETERMINISTIC_NEIGHBOR_PARAMETERS,
                    random_act=True,
                )
        else:
            self.policy.set_training_mode(False)
            with th.no_grad():
                env_action, embed_action = self.policy.wolp_act(
                    self._last_obs,
                    use_target=False,
                    action_noise=action_noise,
                    neighbor_parameters=self.neighbor_parameters,
                    random_act=False,
                )

        assert len(env_action) == 1
        return env_action[0], embed_action[0]

    def train(self, env: AgentEnv, gradient_steps: int, batch_size: int) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        actor_losses, critic_losses = [], []
        for gs in range(gradient_steps):
            logging.debug(
                f"Training agent gradient step {gs}"
            )
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size)

            # Train the critic.
            critic_loss = self.policy.train_critic(
                replay_data, self.neighbor_parameters, self.target_action_noise
            )
            critic_losses.append(critic_loss.item())

            # Train the actor.
            actor_loss = self.policy.train_actor(replay_data)
            actor_losses.append(actor_loss.item())

            self.policy.polyak_update()
