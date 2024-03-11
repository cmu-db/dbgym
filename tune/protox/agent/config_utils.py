from torch import nn
from tune.protox.agent.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


def parse_activation_fn(act_type):
    if act_type == "relu":
        return nn.ReLU
    elif act_type == "gelu":
        return nn.GELU
    elif act_type == "mish":
        return nn.Mish
    elif act_type == "tanh":
        return nn.Tanh
    else:
        assert False, f"Unsupported activation type {act_type}"


def parse_noise_type(noise_type):
    if noise_type == "normal":
        return NormalActionNoise
    elif noise_type == "ou":
        return OrnsteinUhlenbeckActionNoise
    elif noise_type == "none":
        return None
    else:
        assert False, f"Unsupported noise {noise_type}"
