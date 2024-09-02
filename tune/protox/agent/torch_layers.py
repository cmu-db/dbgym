from typing import List, Optional, Type

import torch as th
from torch import nn


def init_layer(
    layer: nn.Module,
    weight_init: Optional[str] = None,
    bias_zero: bool = False,
    final_layer: bool = False,
    final_layer_adjust: float = 1.0,
) -> None:
    if isinstance(layer, nn.Linear):
        if weight_init is not None:
            if weight_init == "orthogonal":
                nn.init.orthogonal_(layer.weight.data)
            elif weight_init == "xavier_uniform":
                nn.init.xavier_uniform_(layer.weight)
            elif weight_init == "xavier_normal":
                nn.init.xavier_normal_(layer.weight)
            else:
                raise ValueError(f"Unknown weight init: {weight_init}")
        if bias_zero:
            layer.bias.data.fill_(0.0)

        if final_layer:
            # Last layer.
            with th.no_grad():
                layer.weight.data = layer.weight.data / final_layer_adjust


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: list[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
    weight_init: Optional[str] = None,
    bias_zero: bool = False,
    final_layer_adjust: float = 1.0,
) -> list[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))

        # Initialize the linear layer weights.
        for layer in modules:
            init_layer(
                layer,
                weight_init,
                bias_zero,
                (layer == modules[-1]),
                final_layer_adjust,
            )

    if squash_output:
        modules.append(nn.Tanh())
    return modules
