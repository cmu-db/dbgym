import numpy as np
import torch
import yaml
from ray import tune

from misc.utils import open_and_save
from tune.protox.agent.buffers import ReplayBuffer
from tune.protox.agent.config_utils import parse_activation_fn, parse_noise_type
from tune.protox.agent.torch_layers import FlattenExtractor
from tune.protox.agent.utils import get_schedule_fn
from tune.protox.agent.wolp.policies import WolpPolicy
from tune.protox.agent.wolp.wolp import Wolp


def setup_wolp_agent(dbgym_cfg, env, spec, seed, agent_config):
    with open_and_save(dbgym_cfg, agent_config, "r") as f:
        config = yaml.safe_load(f)["wolp_params"]

    action_dim = noise_action_dim = env.action_space.get_latent_dim()

    # Get the latent dimension/action space dimension.
    critic_action_dim = env.action_space.get_critic_dim()

    # Setup the noise policy.
    noise_params = config["noise_parameters"]
    means = np.zeros((noise_action_dim,), dtype=np.float32)
    stddevs = np.full(
        (noise_action_dim,), noise_params["noise_sigma"], dtype=np.float32
    )
    noise = parse_noise_type(noise_params["noise_type"])(means, stddevs)

    policy_kwargs = {
        "net_arch": {
            "pi": [int(a) for a in config["pi_arch"].split(",")],
            "qf": [int(a) for a in config["qf_arch"].split(",")],
        },
        "activation_fn": parse_activation_fn(config["activation_fn"]),
        "features_extractor_class": FlattenExtractor,
        "features_extractor_kwargs": None,
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": None,
        "n_critics": 2,
        "share_features_extractor": False,
        "squash_output": False,
        "action_dim": action_dim,
        "critic_action_dim": critic_action_dim,
        "weight_init": config["weight_init"],
        "bias_zero": config["bias_zero"],
        "policy_weight_adjustment": config["policy_weight_adjustment"],
    }

    agent = Wolp(
        WolpPolicy,
        env,
        learning_rate=get_schedule_fn(config["learning_rate"]),
        critic_lr_scale=config["critic_lr_scale"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],
        tau=config["tau"],
        gamma=spec.gamma,
        train_freq=(config["train_freq_frequency"], config["train_freq_unit"]),
        gradient_steps=config["gradient_steps"],
        action_noise=noise,
        replay_buffer_class=ReplayBuffer,
        replay_buffer_kwargs={"action_dim": critic_action_dim},
        target_policy_noise=config["target_policy_noise"],
        target_noise_clip=config["target_noise_clip"],
        policy_kwargs=policy_kwargs,
        seed=seed,
        device="auto",
        _init_setup_model=True,
        neighbor_parameters=config["neighbor_parameters"],
        grad_clip=spec.grad_clip,
        policy_l2_reg=config["policy_l2_reg"],
        noise_action_dim=noise_action_dim,
        epsilon_greedy=config["epsilon_greedy"],
    )
    return agent


def _mutate_wolp_config(dbgym_cfg, hpo_config, protox_args):
    agent_params_path = protox_args.agent_params_path
    with open_and_save(dbgym_cfg, agent_params_path, "r") as f:
        agent_params = yaml.safe_load(f)

    wolp = agent_params["wolp_params"]
    wolp["learning_rate"] = hpo_config.learning_rate
    wolp["critic_lr_scale"] = hpo_config.critic_lr_scale
    wolp["policy_l2_reg"] = hpo_config.policy_l2_reg
    wolp["learning_starts"] = hpo_config.learning_starts
    wolp["tau"] = hpo_config.tau
    wolp["epsilon_greedy"] = hpo_config.epsilon_greedy
    wolp["buffer_size"] = hpo_config.buffer_size
    wolp["batch_size"] = hpo_config.batch_size

    wolp["gradient_steps"] = hpo_config.gradient_steps
    wolp["target_noise_clip"] = hpo_config.target_noise_clip
    wolp["target_policy_noise"] = hpo_config.target_policy_noise

    wolp["train_freq_frequency"] = hpo_config.train_freq_frequency
    wolp["train_freq_unit"] = hpo_config.train_freq_unit

    wolp["noise_parameters"]["noise_type"] = hpo_config.noise_type
    wolp["noise_parameters"]["noise_sigma"] = hpo_config.noise_sigma

    wolp["neighbor_parameters"]["knob_num_nearest"] = hpo_config.knob_num_nearest
    wolp["neighbor_parameters"]["knob_span"] = hpo_config.knob_span
    wolp["neighbor_parameters"]["index_num_samples"] = hpo_config.index_num_samples
    wolp["neighbor_parameters"]["index_subset"] = hpo_config.index_subset

    wolp["weight_init"] = hpo_config.weight_init
    wolp["bias_zero"] = hpo_config.bias_zero
    wolp["policy_weight_adjustment"] = hpo_config.policy_weight_adjustment
    wolp["activation_fn"] = hpo_config.activation_fn
    wolp["pi_arch"] = hpo_config.pi_arch
    wolp["qf_arch"] = hpo_config.qf_arch

    with open("model_params.yaml", "w") as f:
        yaml.dump(agent_params, stream=f, default_flow_style=False)


def _construct_wolp_config():
    return {
        # Learning rate.
        "learning_rate": tune.choice([1e-3, 8e-4, 6e-4, 3e-4, 5e-5, 3e-5, 1e-5]),
        "critic_lr_scale": tune.choice([1.0, 2.5, 5.0, 7.5, 10.0]),
        "policy_l2_reg": tune.choice([0.0, 0.01, 0.03, 0.05]),
        # Number of warmup steps.
        "learning_starts": 0,
        # Polyak averaging rate.
        "tau": tune.choice([1.0, 0.99, 0.995]),
        "epsilon_greedy": 0.0,
        # Replay Buffer Size.
        "buffer_size": 1_000_000,
        # Batch size.
        "batch_size": tune.choice([8, 16, 32, 64]),
        # Gradient steps per sample.
        "gradient_steps": tune.choice([1, 2, 4]),
        # Target noise.
        "target_noise": {
            "target_noise_clip": tune.choice([0, 0.05, 0.1, 0.15]),
            "target_policy_noise": tune.sample_from(
                lambda spc: (
                    0.1
                    if spc["config"]["target_noise"]["target_noise_clip"] == 0
                    else float(np.random.choice([0.05, 0.1, 0.15, 0.2]))
                )
            ),
        },
        # Training steps.
        "train_freq_unit": tune.choice(["step", "step", "episode"]),
        "train_freq_frequency": tune.sample_from(
            lambda spc: (
                1
                if spc["config"]["train_freq_unit"] == "episode"
                else int(np.random.choice([1, 2]))
            )
        ),
        # Noise parameters.
        "noise_parameters": {
            "noise_type": tune.choice(["normal", "ou"]),
            "noise_sigma": tune.choice([0.01, 0.05, 0.1, 0.15, 0.2]),
        },
        "scale_noise_perturb": False,
        # Neighbor parameters.
        "neighbor_parameters": {
            "knob_num_nearest": tune.choice([100, 200]),
            "knob_span": tune.choice([1, 2]),
            "index_num_samples": 1,
            "index_subset": tune.choice([False, True]),
        },
        # LSC Parameters.
        "lsc_parameters": {
            "lsc_enabled": True,
            "lsc_embed": True,
            "lsc_shift_initial": tune.choice(["0,1,2,3,4", "0,0,1,1,1", "0,0,1,1,2"]),
            "lsc_shift_increment": tune.choice(["1", "2", "5", "10"]),
            "lsc_shift_max": tune.choice(["5", "15"]),
            "lsc_shift_schedule_eps_freq": tune.choice([1, 2]),
            "lsc_shift_after": tune.choice([4, 5]),
        },
        # VAE metadata.
        "vae_metadata": {
            "index_vae": True,
            "index_repr": tune.choice(["ONE_HOT_DETERMINISTIC"]),
        },
        "weight_init": tune.choice(["xavier_normal", "xavier_uniform", "orthogonal"]),
        "bias_zero": tune.choice([False, True]),
        "policy_weight_adjustment": tune.choice([1, 100]),
        # Activation
        "activation_fn": tune.choice(["gelu", "mish"]),
        # Architecture.
        "pi_arch": tune.choice(["128", "256", "128,128", "256,256", "512", "256,512"]),
        "qf_arch": tune.choice(
            [
                "256,64",
                "256,256",
                "256,128,128",
                "256,64,64",
                "512",
                "512,256",
                "1024",
                "1024,256",
            ]
        ),
    }
