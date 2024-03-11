import time
import logging
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from tune.protox.agent.buffers import ReplayBuffer
from tune.protox.agent.noise import ActionNoise
from tune.protox.agent.off_policy_algorithm import OffPolicyAlgorithm
from tune.protox.agent.policies import BasePolicy
from tune.protox.agent.type_aliases import GymEnv, Schedule
from tune.protox.agent.utils import polyak_update
from tune.protox.agent.utils import update_learning_rate
from tune.protox.agent.wolp.policies import MlpPolicy, WolpPolicy

SelfWolp = TypeVar("SelfWolp", bound="Wolp")


class Wolp(OffPolicyAlgorithm):
    """
    Wolpertinger DDPG based on Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: The policy model to use (MlpPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use.  If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = { "MlpPolicy": MlpPolicy, }

    accept = [
        "policy_kwargs",
        "num_timesteps",
        "_total_timesteps",
        "_num_timesteps_at_start",
        "seed",
        "action_noise",
        "start_time",
        "_last_obs",
        "_last_episode_starts",
        "_episode_num",
        "_current_progress_remaining",
        "n_updates",
        "replay_buffer",
        "critic_lr_scale",
        "policy_l2_reg",
        "grad_clip",
        "target_noise_clip",
        "target_policy_noise",
        "policy",
    ]

    def save_state(self):
        kv = {}
        state = self.__dict__.keys()
        for k in state:
            if k in self.accept:
                if k in ["policy", "replay_buffer"]:
                    kv[k] = getattr(self, k).save_state()
                else:
                    kv[k] = getattr(self, k)
        return kv

    def load_state(self, d):
        for k in self.accept:
            if k in d:
                if k in ["policy", "replay_buffer"]:
                    getattr(self, k).load_state(d[k])
                else:
                    setattr(self, k, d[k])

    def __init__(
        self,
        policy: Union[str, Type[WolpPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        critic_lr_scale = 1.0,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        neighbor_parameters: Dict[str, Any] = {},
        grad_clip: float = 1.0,
        policy_l2_reg: float = 0.0,
        noise_action_dim: int = 0,
        epsilon_greedy: float = 0.0,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            device=device,
            seed=seed,
            supported_action_spaces=(spaces.Tuple),
            support_multi_env=True,
        )

        self.critic_lr_scale = critic_lr_scale
        self.policy_l2_reg = policy_l2_reg
        self.grad_clip = grad_clip
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.neighbor_parameters = neighbor_parameters
        self.noise_action_dim = noise_action_dim
        self.epsilon_greedy = epsilon_greedy

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
        log_latents: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
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
                deterministic_neighbor_parameters = {
                    "knob_num_nearest": 1,
                    "knob_span": 0,
                    "index_num_samples": 1,
                    "index_subset": True,
                }

                env_action, embed_action = self.policy.wolp_act(
                    self._last_obs,
                    use_target=False,
                    action_noise=None,
                    neighbor_parameters=deterministic_neighbor_parameters,
                    instr_logger=self.logger,
                    random_act=True,
                )
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            noise = None
            if action_noise is not None:
                noise = action_noise

            self.policy.set_training_mode(False)
            with th.no_grad():
                env_action, embed_action = self.policy.wolp_act(
                    self._last_obs,
                    use_target=False,
                    action_noise=noise,
                    neighbor_parameters=self.neighbor_parameters,
                    instr_logger=self.logger,
                    random_act=False,
                    epsilon_greedy=self.epsilon_greedy,
                )

        return env_action, embed_action

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        start = time.time()
        policy_loss_steps = 0

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        update_learning_rate(self.critic.optimizer, self.critic_lr_scale * self.lr_schedule(self._current_progress_remaining))

        actor_losses, critic_losses = [], []
        l2_losses, actor_grad_losses = [], []
        for gs in range(gradient_steps):
            logging.debug(f"Training agent gradient step {gs}")
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                def noise_fn():
                    means = th.zeros((replay_data.actions.shape[0], self.noise_action_dim,), dtype=th.float32)
                    return th.normal(means, self.target_policy_noise).clamp(-self.target_noise_clip, self.target_noise_clip).float()

                # wolp_act() actually gives both the env and the embedding actions.
                # We evaluate the critic on the embedding action and not the environment action.
                _, embed_actions = self.policy.wolp_act(
                    replay_data.next_observations,
                    use_target=True,
                    action_noise=noise_fn,
                    neighbor_parameters=self.neighbor_parameters)
                embed_actions = th.as_tensor(embed_actions, device=self.policy.device).float()

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, embed_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get the current action representation.
            wolp_start = time.time()
            embeds = replay_data.actions.float()

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, embeds)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            assert not th.isnan(critic_loss).any()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(list(self.critic.parameters()), self.grad_clip)
            self.critic.optimizer.step()

            # Compute actor loss
            raw_actions = self.actor(replay_data.observations)
            if (not hasattr(self, "_ignore_index")) or (not self._ignore_index):
                lscs = replay_data.lscs.reshape(-1, 1)
                raw_actions = self.env.action_space.adjust_action_lsc(raw_actions, lscs)

                if self.env.action_space.get_index_space().index_space_aux_type_dim > 0:
                    raw_actions = th.concat([embeds[:, :self.env.action_space.get_index_space().index_space_aux_type_dim], raw_actions], dim=1)
                if self.env.action_space.get_index_space().index_space_aux_include > 0:
                    raw_actions = th.concat([raw_actions, embeds[:, -self.env.action_space.get_index_space().index_space_aux_include:]], dim=1)

            actor_loss = -self.critic.q1_forward(replay_data.observations, raw_actions).mean()
            actor_grad_losses.append(actor_loss.item())

            # Attach l2.
            l2_loss = 0
            if self.policy_l2_reg > 0:
                for param in self.actor.parameters():
                    l2_loss += 0.5 * (param ** 2).sum()
                l2_losses.append(l2_loss.item())
            actor_loss += l2_loss
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            assert not th.isnan(actor_loss).any()
            actor_loss.backward()

            th.nn.utils.clip_grad_norm_(list(self.actor.parameters()), self.grad_clip)
            self.actor.optimizer.step()

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
            policy_loss_steps += 1

        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/actor_loss", 0 if len(actor_losses) == 0 else np.mean(actor_losses))
        self.logger.record("train/actor_l2_loss", 0 if len(l2_losses) == 0 else np.mean(l2_losses))
        self.logger.record("train/actor_grad_loss", 0 if len(actor_grad_losses) == 0 else np.mean(actor_grad_losses))
        self.logger.record("train/critic_loss", 0 if len(critic_losses) == 0 else np.mean(critic_losses))

        # Log as munch time metrics as we can too.
        wolp_time = time.time() - start
        self.logger.record("instr_time/learn_wolp_time", wolp_time)

    def learn(
        self: SelfWolp,
        total_timesteps: int,
        log_interval: int = 4,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfWolp:
        return super().learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
