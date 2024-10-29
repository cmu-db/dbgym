from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from tune.protox.agent.agent_env import AgentEnv
from tune.protox.agent.base_class import BaseAlgorithm
from tune.protox.agent.buffers import ReplayBuffer
from tune.protox.agent.noise import ActionNoise
from tune.protox.agent.utils import (
    RolloutReturn,
    TrainFreq,
    TrainFrequencyUnit,
    should_collect_more_steps,
)
from util.workspace import TuningMode


class OffPolicyAlgorithm(BaseAlgorithm):
    """
    The base for Off-Policy algorithms (ex: SAC/TD3)

    :param policy: The policy model
    :param replay_buffer
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param seed: Seed for the pseudo random generators
    """

    def __init__(
        self,
        policy: Any,
        replay_buffer: ReplayBuffer,
        learning_starts: int = 100,
        batch_size: int = 256,
        train_freq: tuple[int, str] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        seed: Optional[int] = None,
        ray_trial_id: Optional[str] = None,
    ):
        super().__init__(seed=seed)
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.ray_trial_id = ray_trial_id

        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.gradient_steps = gradient_steps
        self.action_noise = action_noise

        # Save train freq parameter, will be converted later to TrainFreq object
        self.train_freq = self._convert_train_freq(train_freq)

    def _convert_train_freq(self, train_freq: tuple[int, str]) -> TrainFreq:
        """
        Convert `train_freq` parameter (int or tuple)
        to a TrainFreq object.
        """
        return TrainFreq(*(train_freq[0], TrainFrequencyUnit(train_freq[1])))

    def train(self, env: AgentEnv, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        raise NotImplementedError()

    def _on_step(self) -> None:
        """
        Method called after each step in the environment.
        It is meant to trigger DQN target network update
        but can be used for other purposes
        """
        pass

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
        raise NotImplementedError()

    def collect_rollouts(
        self,
        tuning_mode: TuningMode,
        env: AgentEnv,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, AgentEnv), "You must pass a AgentEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        continue_training = True
        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            if self.timeout_checker is not None and self.timeout_checker():
                # Timeout has been hit.
                continue_training = False
                break

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise)

            # Rescale and perform action
            new_obs, rewards, terms, truncs, infos = env.step(actions)
            dones = terms or truncs
            # We only stash the results if we're not doing HPO, or else the results from concurrent HPO would get
            #   stashed in the same directory and potentially cause a race condition.
            if self.artifact_manager:
                assert (
                    self.ray_trial_id != None if tuning_mode == TuningMode.HPO else True
                ), "If we're doing HPO, we need to ensure that we're passing a non-None ray_trial_id to stash_results() to avoid conflicting folder names."
                self.artifact_manager.stash_results(
                    infos, ray_trial_id=self.ray_trial_id
                )

            self.num_timesteps += 1
            num_collected_steps += 1

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(
                replay_buffer, buffer_actions, new_obs, rewards, dones, infos
            )

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            if dones:
                # Update stats
                num_collected_episodes += 1
                self._episode_num += 1
                if action_noise is not None:
                    action_noise.reset()

        return RolloutReturn(
            num_collected_steps, num_collected_episodes, continue_training
        )

    def learn(
        self, env: AgentEnv, total_timesteps: int, tuning_mode: TuningMode
    ) -> None:
        assert isinstance(env, AgentEnv)
        total_timesteps = self._setup_learn(env, total_timesteps)

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                tuning_mode,
                env,
                train_freq=self.train_freq,
                replay_buffer=self.replay_buffer,
                action_noise=self.action_noise,
                learning_starts=self.learning_starts,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = (
                    self.gradient_steps
                    if self.gradient_steps >= 0
                    else rollout.episode_timesteps
                )
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(
                        env,
                        gradient_steps=gradient_steps,
                        batch_size=self.batch_size,
                    )
