import logging
import time
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from tune.protox.agent.policies import BasePolicy, ContinuousCritic
from tune.protox.agent.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
    get_actor_critic_arch,
)
from tune.protox.agent.type_aliases import Schedule


class Actor(BasePolicy):
    """
    Actor network (policy) for wolpertinger architecture (based on TD3).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features (a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        weight_init=None,
        bias_zero=False,
        squash_output: bool = False,
        action_dim: int = 0,
        policy_weight_adjustment=1,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            squash_output=squash_output,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.action_dim = action_dim
        assert action_dim > 0

        actor_net = create_mlp(
            features_dim,
            action_dim,
            net_arch,
            activation_fn,
            squash_output=squash_output,
            weight_init=weight_init,
            bias_zero=bias_zero,
            final_layer_adjust=policy_weight_adjustment,
        )
        # Deterministic action
        self.mu = nn.Sequential(*actor_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs, self.features_extractor)
        return self.action_space.process_network_output(self.mu(features))

    def _predict(
        self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self(observation)

    def save_state(self):
        return {
            "mu": self.mu,
        }

    def load_state(self, d):
        self.mu = d["mu"]


class WolpPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for Wolp.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        squash_output: bool = False,
        action_dim: int = 0,
        critic_action_dim: int = 0,
        weight_init=None,
        bias_zero=False,
        policy_weight_adjustment=1,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from the original paper
        if net_arch is None:
            net_arch = [400, 300]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "weight_init": weight_init,
            "bias_zero": bias_zero,
        }
        self.actor_kwargs = self.net_args.copy()
        self.actor_kwargs.update(
            {
                "squash_output": squash_output,
                "action_dim": action_dim,
                "policy_weight_adjustment": policy_weight_adjustment,
            }
        )

        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
                "action_dim": critic_action_dim,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def save_state(self):
        return {
            "actor": self.actor.save_state(),
            "critic": self.critic.save_state(),
            "critic_target": self.critic_target.save_state(),
            "actor_optimizer": self.actor.optimizer,
            "critic_optimizer": self.critic.optimizer,
        }

    def load_state(self, d):
        self.actor.load_state(d["actor"])
        self.critic.load_state(d["critic"])
        self.critic_target.load_state(d["critic_target"])
        self.actor.optimizer = d["actor_optimizer"]
        self.critic.optimizer = d["critic_optimizer"]

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(
                features_extractor=self.actor.features_extractor
            )
            # Critic target should not share the features extractor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_target = self.make_critic(
                features_extractor=self.actor_target.features_extractor
            )
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

        # Log all the networks.
        logging.info("Actor: %s", self.actor)
        logging.info("Critic: %s", self.critic)

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        assert False

    def wolp_act(
        self,
        states,
        use_target: bool = False,
        action_noise=None,
        neighbor_parameters=None,
        instr_logger=None,
        random_act=False,
        epsilon_greedy=0.0,
    ):
        # Get the tensor representation.
        start_time = time.time()
        if not isinstance(states, th.Tensor):
            states, _ = self.obs_to_tensor(states)

        if random_act:
            raw_action = self.action_space.random_embed_action()
            raw_action = th.as_tensor(raw_action, device=self.device).float()
        elif use_target:
            raw_action = self.actor_target(states)
        else:
            raw_action = self.actor(states)

        if action_noise is not None:
            assert not random_act
            noise = th.as_tensor(action_noise(), device=self.device).float()
            if len(noise.shape) == 1:
                # Reshape into single-sample tensor.
                noise = noise.reshape(1, -1)

            # Apply the noise.
            raw_action = self.action_space.perturb_noise(raw_action, noise)

        # Smear the action.
        random = (epsilon_greedy > 0.0) and (np.random.rand() < epsilon_greedy)
        env_actions, sample_actions, actions_dim = self.action_space.actor_smear_action(
            raw_action, neighbor_parameters, random=random
        )

        if random_act:
            # If we want a random action, don't use Q-value estimate.
            rand_act = np.random.randint(0, high=len(env_actions))
            if instr_logger is not None:
                instr_logger.record("instr_time/next_act", time.time() - start_time)

            return [env_actions[rand_act]], sample_actions[rand_act : rand_act + 1]

        assert states.shape[0] == actions_dim.shape[0]
        assert len(states.shape) == 2
        env_actions, embed_actions, _, raw_qs = self.discriminate(
            use_target, states, sample_actions, actions_dim, env_actions
        )

        if instr_logger is not None:
            instr_logger.record("instr_time/next_act", time.time() - start_time)
        assert not np.isnan(embed_actions).any()
        return env_actions, embed_actions

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


MlpPolicy = WolpPolicy
