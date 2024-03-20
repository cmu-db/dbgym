import os
import socket
import glob
from pathlib import Path
import shutil
import json
import xml.etree.ElementTree as ET
import torch
from torch import nn
import numpy as np
import gymnasium as gym
from typing import Callable, Tuple, Any, Union

from tune.protox.env.logger import Logger
from tune.protox.env.util.reward import RewardUtility
from tune.protox.env.util.postgres import PostgresConn
from tune.protox.env.workload import Workload
from tune.protox.env.space.holon_space import HolonSpace
from tune.protox.env.space.state.space import StateSpace
from tune.protox.env.space.state import LSCStructureStateSpace, LSCMetricStateSpace
from tune.protox.env.space.latent_space.latent_knob_space import LatentKnobSpace
from tune.protox.env.lsc.lsc import LSC
from tune.protox.env.space.latent_space.lsc_index_space import LSCIndexSpace
from tune.protox.env.space.latent_space.latent_query_space import LatentQuerySpace
from tune.protox.env.lsc.lsc_wrapper import LSCWrapper
from tune.protox.env.mqo.mqo_wrapper import MQOWrapper
from tune.protox.env.target_reset.target_reset_wrapper import TargetResetWrapper
from tune.protox.agent.agent_env import AgentEnv
from tune.protox.agent.utils import parse_noise_type
from tune.protox.agent.wolp.wolp import Wolp
from tune.protox.agent.wolp.policies import WolpPolicy
from tune.protox.agent.policies import ContinuousCritic, Actor
from gymnasium.wrappers import NormalizeObservation, NormalizeReward, FlattenObservation # type: ignore
from tune.protox.agent.buffers import ReplayBuffer
from tune.protox.env.target_reset.target_reset_wrapper import TargetResetWrapper
from tune.protox.agent.noise import ClampNoise
from tune.protox.embedding.train_all import create_vae_model , fetch_vae_parameters_from_workload
from tune.protox.env.types import TableAttrAccessSetsMap, ProtoAction


def _parse_activation_fn(act_type: str) -> type[nn.Module]:
    if act_type == "relu":
        return nn.ReLU
    elif act_type == "gelu":
        return nn.GELU
    elif act_type == "mish":
        return nn.Mish
    elif act_type == "tanh":
        return nn.Tanh
    else:
        raise ValueError(f"Unsupported activation type {act_type}")


def _get_signal(signal_folder: Union[str, Path]) -> Tuple[int, str]:
    MIN_PORT = 5434
    MAX_PORT = 5500

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = MIN_PORT
    while port <= MAX_PORT:
        try:
            s.bind(('', port))

            drop = False
            for sig in glob.glob(f"{signal_folder}/*.signal"):
                if port == int(Path(sig).stem):
                    drop = True
                    break

            # Someone else has actually taken hold of this.
            if drop:
                port += 1
                s.close()
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                continue

            with open(f"{signal_folder}/{port}.signal", "w") as f: # type: IO[Any]
                f.write(str(port))
                f.close()

            s.close()
            return port, f"{signal_folder}/{port}.signal"
        except OSError as e:
            port += 1
    raise IOError("No free ports to bind postgres to.")


def _edit_args(logdir: str, port: int, hpo_config: dict[str, Any]) -> dict[str, Any]:
    mythril_dir = hpo_config["mythril_dir"]
    if hpo_config["benchmark_config"]["query_spec"]["query_directory"][0] != "/":
        hpo_config["benchmark_config"]["query_spec"]["query_directory"] = mythril_dir + "/" + hpo_config["benchmark_config"]["query_spec"]["query_directory"]
    if hpo_config["benchmark_config"]["query_spec"]["query_order"][0] != "/":
        hpo_config["benchmark_config"]["query_spec"]["query_order"] = mythril_dir + "/" + hpo_config["benchmark_config"]["query_spec"]["query_order"]

    if "execute_query_directory" in hpo_config["benchmark_config"]["query_spec"]:
        if hpo_config["benchmark_config"]["query_spec"]["execute_query_directory"][0] != "/":
            hpo_config["benchmark_config"]["query_spec"]["execute_query_directory"] = mythril_dir + "/" + hpo_config["benchmark_config"]["query_spec"]["execute_query_directory"]
        if hpo_config["benchmark_config"]["query_spec"]["execute_query_order"][0] != "/":
            hpo_config["benchmark_config"]["query_spec"]["execute_query_order"] = mythril_dir + "/" + hpo_config["benchmark_config"]["query_spec"]["execute_query_order"]

    if "benchbase_config_path" in hpo_config["benchbase_config"]:
        # Copy the benchbase benchmark path.
        shutil.copy(hpo_config["benchbase_config"]["benchbase_config_path"], Path(logdir) / "benchmark.xml")
        hpo_config["benchbase_config"]["benchbase_config_path"] = Path(logdir) / "benchmark.xml"

    if "benchbase_path" in hpo_config["benchbase_config"]:
        hpo_config["benchbase_config"]["benchbase_path"] = os.path.expanduser(hpo_config["benchbase_config"]["benchbase_path"])

    # Append port to the pgdata folder.
    hpo_config["pgconn"]["pg_data"] += f"{port}"

    kvs = {s.split("=")[0]: s.split("=")[1] for s in hpo_config["pgconn"]["pg_conn"].split(" ")}
    kvs["port"] = str(port)
    hpo_config["pgconn"]["pg_conn"] = " ".join([f"{k}={v}" for k, v in kvs.items()])

    if hpo_config["benchmark_config"]["query_spec"]["oltp_workload"]:
        conf_etree = ET.parse(Path(logdir) / "benchmark.xml")
        jdbc = f"jdbc:postgresql://localhost:{port}/benchbase?preferQueryMode=extended"
        conf_etree.getroot().find("url").text = jdbc # type: ignore

        oltp_config = hpo_config["benchbase_config"]["oltp_config"]
        if conf_etree.getroot().find("scalefactor") is not None:
            conf_etree.getroot().find("scalefactor").text = str(oltp_config["oltp_sf"]) # type: ignore
        if conf_etree.getroot().find("terminals") is not None:
            conf_etree.getroot().find("terminals").text = str(oltp_config["oltp_num_terminals"]) # type: ignore
        if conf_etree.getroot().find("works") is not None:
            works = conf_etree.getroot().find("works").find("work") # type: ignore
            if works.find("time") is not None: # type: ignore
                conf_etree.getroot().find("works").find("work").find("time").text = str(oltp_config["oltp_duration"]) # type: ignore
            if works.find("warmup") is not None: # type: ignore
                conf_etree.getroot().find("works").find("work").find("warmup").text = str(oltp_config["oltp_warmup"]) # type: ignore
        conf_etree.write(Path(logdir) / "benchmark.xml")

    class Encoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            return super(Encoder, self).default(obj)

    with open(Path(logdir) / "hpo_params.json", "w") as f:
        json.dump(hpo_config, f, indent=4, cls=Encoder)

    return hpo_config


def _gen_noise_scale(vae_config: dict[str, Any], hpo_config: dict[str, Any]) -> Callable[[ProtoAction, torch.Tensor], ProtoAction]:
    def f(p: ProtoAction, n: torch.Tensor) -> ProtoAction:
        if hpo_config["scale_noise_perturb"]:
            return ProtoAction(torch.clamp(p + n * vae_config["output_scale"], 0., vae_config["output_scale"]))
        else:
            return ProtoAction(torch.clamp(p + n, 0., 1.))
    return f


def _build_utilities(logdir: str, hpo_config: dict[str, Any]) -> Tuple[Logger, RewardUtility, PostgresConn, Workload]:
    logger = Logger(
        hpo_config["trace"],
        hpo_config["verbose"],
        Path(logdir) / hpo_config["output_log_path"],
        Path(logdir) / hpo_config["output_log_path"] / "repository",
        Path(logdir) / hpo_config["output_log_path"] / "tboard",
    )

    reward_utility = RewardUtility(
        target="tps" if hpo_config["benchmark_config"]["query_spec"]["oltp_workload"] else "latency",
        metric=hpo_config["reward"],
        reward_scaler=hpo_config["reward_scaler"],
        logger=logger,
    )

    pgconn = PostgresConn(
        postgres_conn=hpo_config["pgconn"]["pg_conn"],
        postgres_data=hpo_config["pgconn"]["pg_data"],
        postgres_path=hpo_config["pgconn"]["pg_bins"],
        postgres_logs_dir=Path(logdir) / hpo_config["output_log_path"] / "pg_logs",
        connect_timeout=300,
        logger=logger,
    )

    workload = Workload(
        tables=hpo_config["benchmark_config"]["tables"],
        attributes=hpo_config["benchmark_config"]["attributes"],
        query_spec=hpo_config["benchmark_config"]["query_spec"],
        pid=None,
        workload_timeout=hpo_config["workload_timeout"],
        workload_timeout_penalty=hpo_config["workload_timeout_penalty"],
        logger=logger,
    )

    return logger, reward_utility, pgconn, workload


def _build_actions(seed: int, hpo_config: dict[str, Any], workload: Workload, logger: Logger) -> Tuple[HolonSpace, LSC]:
    sysknobs = LatentKnobSpace(
        logger=logger,
        tables=hpo_config["benchmark_config"]["tables"],
        knobs=hpo_config["system_knobs"],
        quantize=True,
        quantize_factor=hpo_config["default_quantization_factor"],
        seed=seed,
        table_level_knobs=hpo_config["benchmark_config"]["table_level_knobs"],
        latent=True,
    )

    with open(Path(hpo_config["embeddings"]).parent / "config") as f:
        vae_config = json.load(f)

        assert vae_config["mean_output_act"] == "sigmoid"
        index_output_transform = lambda x: torch.nn.Sigmoid()(x) * vae_config["output_scale"]
        index_noise_scale = _gen_noise_scale(vae_config, hpo_config)

        max_attrs, max_cat_features = fetch_vae_parameters_from_workload(workload, len(hpo_config["benchmark_config"]["tables"]))
        vae = create_vae_model(vae_config, max_attrs, max_cat_features)
        vae.load_state_dict(torch.load(hpo_config["embeddings"]))

    lsc = LSC(
        horizon=hpo_config["horizon"],
        lsc_parameters=hpo_config["lsc"],
        vae_config=vae_config,
        logger=logger,
    )

    idxspace = LSCIndexSpace(
        tables=hpo_config["benchmark_config"]["tables"],
        max_num_columns=hpo_config["benchmark_config"]["max_num_columns"],
        max_indexable_attributes=workload.max_indexable(),
        seed=seed,
        # TODO(wz2): We should theoretically pull this from the DBMS.
        rel_metadata=hpo_config["benchmark_config"]["attributes"],
        attributes_overwrite=workload.column_usages(),
        tbl_include_subsets=TableAttrAccessSetsMap(workload.tbl_include_subsets),
        index_space_aux_type=hpo_config["benchmark_config"]["index_space_aux_type"],
        index_space_aux_include=hpo_config["benchmark_config"]["index_space_aux_include"],
        deterministic_policy=True,
        vae=vae,
        latent_dim=vae_config["latent_dim"],
        index_output_transform=index_output_transform,
        index_noise_scale=index_noise_scale,
        logger=logger,
        lsc=lsc,
    )

    qspace = LatentQuerySpace(
        tables=hpo_config["benchmark_config"]["tables"],
        quantize=True,
        quantize_factor=hpo_config["default_quantization_factor"],
        seed=seed,
        per_query_knobs_gen=hpo_config["benchmark_config"]["per_query_knob_gen"],
        per_query_parallel={} if not hpo_config["benchmark_config"]["per_query_select_parallel"] else workload.query_aliases,
        per_query_scans={} if not hpo_config["benchmark_config"]["per_query_scan_method"] else workload.query_aliases,
        query_names=workload.order,
        logger=logger,
        latent=True,
    )

    hspace = HolonSpace(
        knob_space=sysknobs,
        index_space=idxspace,
        query_space=qspace,
        seed=seed,
        logger=logger,
    )
    return hspace, lsc


def _build_obs_space(action_space: HolonSpace, lsc: LSC, hpo_config: dict[str, Any], seed: int) -> StateSpace:
    if hpo_config["metric_state"] == "metric":
        return LSCMetricStateSpace(
            lsc=lsc,
            tables=hpo_config["benchmark_config"]["tables"],
            seed=seed,
        )
    elif hpo_config["metric_state"] == "structure":
        return LSCStructureStateSpace(
            lsc=lsc,
            action_space=action_space,
            seed=seed,
        )
    else:
        ms = hpo_config["metric_state"]
        raise ValueError(f"Unsupported state representation {ms}")


def _build_env(
        hpo_config: dict[str, Any],
        pgconn: PostgresConn,
        obs_space: StateSpace,
        holon_space: HolonSpace,
        lsc: LSC,
        workload: Workload,
        reward_utility: RewardUtility,
        logger: Logger) -> Tuple[TargetResetWrapper, AgentEnv]:

    env = gym.make("Postgres-v0",
        observation_space=obs_space,
        action_space=holon_space,
        workload=workload,
        data_snapshot_path=hpo_config["data_snapshot_path"],
        horizon=hpo_config["horizon"],
        reward_utility=reward_utility,
        pgconn=pgconn,
        pqt=hpo_config["query_timeout"],
        benchbase_config=hpo_config["benchbase_config"],
        logger=logger,
        replay=False,
    )

    # Check whether to create the MQO wrapper.
    if not hpo_config["benchmark_config"]["query_spec"]["oltp_workload"]:
        if hpo_config["workload_eval_mode"] != "pq" or hpo_config["workload_eval_inverse"] or hpo_config["workload_eval_reset"]:
            env = MQOWrapper(
                workload_eval_mode=hpo_config["workload_eval_mode"],
                workload_eval_inverse=hpo_config["workload_eval_inverse"],
                workload_eval_reset=hpo_config["workload_eval_reset"],
                benchbase_config=hpo_config["benchbase_config"],
                pqt=hpo_config["query_timeout"],
                env=env,
                logger=logger,
            )

    # Attach LSC.
    env = LSCWrapper(
        lsc=lsc,
        env=env,
        logger=logger,
    )

    # Attach TargetResetWrapper.
    target_reset = env = TargetResetWrapper(
        env=env,
        maximize_state=hpo_config["maximize_state"],
        reward_utility=reward_utility,
        start_reset=False,
        logger=logger,
    )

    env = FlattenObservation(env)
    if hpo_config["normalize_state"]:
        env = NormalizeObservation(env)

    if hpo_config["normalize_reward"]:
        env = NormalizeReward(env, gamma=hpo_config["gamma"])

    # Wrap the AgentEnv to have null checking.
    env = AgentEnv(env)
    return target_reset, env


def _build_agent(seed: int, hpo_config: dict[str, Any], obs_space: StateSpace, action_space: HolonSpace, logger: Logger) -> Wolp:
    action_dim = noise_action_dim = action_space.latent_dim()
    critic_action_dim = action_space.critic_dim()

    actor = Actor(
        observation_space=obs_space,
        action_space=action_space,
        net_arch=[int(l) for l in hpo_config["pi_arch"].split(",")],
        features_dim=gym.spaces.utils.flatdim(obs_space),
        activation_fn=_parse_activation_fn(hpo_config["activation_fn"]),
        weight_init=hpo_config["weight_init"],
        bias_zero=hpo_config["bias_zero"],
        squash_output=False,
        action_dim=action_dim,
        policy_weight_adjustment=hpo_config["policy_weight_adjustment"],
    )

    actor_target = Actor(
        observation_space=obs_space,
        action_space=action_space,
        net_arch=[int(l) for l in hpo_config["pi_arch"].split(",")],
        features_dim=gym.spaces.utils.flatdim(obs_space),
        activation_fn=_parse_activation_fn(hpo_config["activation_fn"]),
        weight_init=hpo_config["weight_init"],
        bias_zero=hpo_config["bias_zero"],
        squash_output=False,
        action_dim=action_dim,
        policy_weight_adjustment=hpo_config["policy_weight_adjustment"],
    )

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=hpo_config["learning_rate"])

    critic = ContinuousCritic(
        observation_space=obs_space,
        action_space=action_space,
        net_arch=[int(l) for l in hpo_config["qf_arch"].split(",")],
        features_dim=gym.spaces.utils.flatdim(obs_space),
        activation_fn=_parse_activation_fn(hpo_config["activation_fn"]),
        weight_init=hpo_config["weight_init"],
        bias_zero=hpo_config["bias_zero"],
        n_critics=2,
        action_dim=critic_action_dim,
    )

    critic_target = ContinuousCritic(
        observation_space=obs_space,
        action_space=action_space,
        net_arch=[int(l) for l in hpo_config["qf_arch"].split(",")],
        features_dim=gym.spaces.utils.flatdim(obs_space),
        activation_fn=_parse_activation_fn(hpo_config["activation_fn"]),
        weight_init=hpo_config["weight_init"],
        bias_zero=hpo_config["bias_zero"],
        n_critics=2,
        action_dim=critic_action_dim,
    )

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=hpo_config["learning_rate"] * hpo_config["critic_lr_scale"])

    policy = WolpPolicy(
        observation_space=obs_space,
        action_space=action_space,
        actor=actor,
        actor_target=actor_target,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_target=critic_target,
        critic_optimizer=critic_optimizer,
        grad_clip=hpo_config["grad_clip"],
        policy_l2_reg=hpo_config["policy_l2_reg"],
        tau=hpo_config["tau"],
        gamma=hpo_config["gamma"],
        logger=logger,
    )

    # Setup the noise policy.
    noise_params = hpo_config["noise_parameters"]
    means = np.zeros((noise_action_dim,), dtype=np.float32)
    stddevs = np.full((noise_action_dim,), noise_params["noise_sigma"], dtype=np.float32)
    action_noise_type = parse_noise_type(noise_params["noise_type"])
    action_noise = None if not action_noise_type else action_noise_type(means, stddevs)

    target_noise = hpo_config["target_noise"]
    means = np.zeros((hpo_config["batch_size"], noise_action_dim,), dtype=np.float32)
    stddevs = np.full((hpo_config["batch_size"], noise_action_dim,), target_noise["target_policy_noise"], dtype=np.float32)
    target_action_noise = parse_noise_type("normal")
    assert target_action_noise
    clamp_noise = ClampNoise(target_action_noise(means, stddevs), target_noise["target_noise_clip"])

    return Wolp(
        policy=policy,
        replay_buffer=ReplayBuffer(
            buffer_size=hpo_config["buffer_size"],
            obs_shape=[gym.spaces.utils.flatdim(obs_space)],
            action_dim=critic_action_dim,
        ),
        learning_starts=hpo_config["learning_starts"],
        batch_size=hpo_config["batch_size"],
        train_freq=(hpo_config["train_freq_frequency"], hpo_config["train_freq_unit"]),
        gradient_steps=hpo_config["gradient_steps"],
        action_noise=action_noise,
        target_action_noise=clamp_noise,
        seed=seed,
        neighbor_parameters=hpo_config["neighbor_parameters"],
    )


def build_trial(seed: int, logdir: str, hpo_config: dict[str, Any]) -> Tuple[Logger, TargetResetWrapper, AgentEnv, Wolp, str]:
    # The massive trial builder.

    port, signal = _get_signal(hpo_config["pgconn"]["pg_bins"])
    hpo_config = _edit_args(logdir, port, hpo_config)

    logger, reward_utility, pgconn, workload = _build_utilities(logdir, hpo_config)
    holon_space, lsc = _build_actions(seed, hpo_config, workload, logger)
    obs_space = _build_obs_space(holon_space, lsc, hpo_config, seed)
    target_reset, env = _build_env(hpo_config, pgconn, obs_space, holon_space, lsc, workload, reward_utility, logger)

    agent = _build_agent(seed, hpo_config, obs_space, holon_space, logger)
    return logger, target_reset, env, agent, signal
