"""Policies: abstract base class and concrete implementations."""

import logging
from typing import Any, List, Optional, Tuple, Type, cast

import numpy as np
import torch as th
from gymnasium import spaces
from numpy.typing import NDArray
from torch import nn

from tune.protox.agent.torch_layers import create_mlp


class BaseModel(nn.Module):
    """
    The base model object: makes predictions in response to observations.

    In the case of policies, the prediction is an action. In the case of critics, it is the
    estimated value of the observation.

    :param observation_space: The observation space of the environment
    :param action_space: The action space of the environment
    :param optimizer_class: The optimizer to use, ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments, excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self, observation_space: spaces.Space[Any], action_space: spaces.Space[Any]
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        return cast(th.Tensor, nn.Flatten()(obs.float()))

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)

    def obs_to_tensor(
        self,
        observation: NDArray[np.float32],
    ) -> th.Tensor:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.
        Includes sugar-coating to handle different observations.

        :param observation: the input observation
        :return: The observation as PyTorch tensor
        """
        observation = np.array(observation)
        if not isinstance(observation, dict):
            # Add batch dimension if needed
            sh = (
                self.observation_space.shape
                if self.observation_space.shape
                else [spaces.utils.flatdim(self.observation_space)]
            )
            observation = observation.reshape((-1, *sh))

        return th.as_tensor(observation)


class Actor(BaseModel):
    """
    Actor network (policy) for wolpertinger architecture (based on TD3).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_dim: Number of features
    :param activation_fn: Activation function
    """

    def __init__(
        self,
        observation_space: spaces.Space[Any],
        action_space: spaces.Space[Any],
        net_arch: list[int],
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        weight_init: Optional[str] = None,
        bias_zero: bool = False,
        squash_output: bool = False,
        action_dim: int = 0,
        policy_weight_adjustment: float = 1.0,
    ):
        super().__init__(observation_space, action_space)

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
        return cast(th.Tensor, self.mu(self.extract_features(obs)))

    def _predict(
        self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return cast(th.Tensor, self(observation))

    def check_grad(self) -> None:
        layers = [l for l in self.mu.children()]
        assert isinstance(layers[0], nn.Linear), f"{layers[0]} {type(layers[0])}"
        assert layers[0].weight.grad is not None
        assert not layers[0].weight.grad.isnan().any()  # Check no NaN.
        assert layers[0].weight.grad.any()  # Check grad at least flows.


class ContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param n_critics: Number of critic networks to create.
    """

    def __init__(
        self,
        observation_space: spaces.Space[Any],
        action_space: spaces.Space[Any],
        net_arch: list[int],
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        weight_init: Optional[str] = None,
        bias_zero: bool = False,
        n_critics: int = 2,
        action_dim: int = 0,
    ):
        super().__init__(observation_space, action_space)

        assert action_dim > 0
        self.action_dim = action_dim
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = nn.Sequential(
                *create_mlp(
                    features_dim + action_dim,
                    1,
                    net_arch,
                    activation_fn,
                    weight_init=weight_init,
                    bias_zero=bias_zero,
                )
            )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        with th.set_grad_enabled(True):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs)
        return cast(th.Tensor, self.q_networks[0](th.cat([features, actions], dim=1)))

    def check_grad(self) -> None:
        layers = [l for l in self.q_networks[0].children()]
        assert isinstance(layers[0], nn.Linear), f"{layers[0]} {type(layers[0])}"
        assert layers[0].weight.grad is not None
        assert not layers[0].weight.grad.isnan().any()  # Check no NaN.
        assert layers[0].weight.grad.any()  # Check grad at least flows.
