import glob
import json
import os
import shutil
import socket
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import (  # type: ignore
    FlattenObservation,
    NormalizeObservation,
    NormalizeReward,
)
from torch import nn

from misc.utils import DBGymConfig, open_and_save, make_redis_started, save_file
from tune.protox.agent.agent_env import AgentEnv
from tune.protox.agent.buffers import ReplayBuffer
from tune.protox.agent.noise import ClampNoise
from tune.protox.agent.policies import Actor, ContinuousCritic
from tune.protox.agent.utils import parse_noise_type
from tune.protox.agent.wolp.policies import WolpPolicy
from tune.protox.agent.wolp.wolp import Wolp
from tune.protox.embedding.train_all import (
    create_vae_model,
    fetch_vae_parameters_from_workload,
)
from tune.protox.env.logger import Logger
from tune.protox.env.lsc.lsc import LSC
from tune.protox.env.lsc.lsc_wrapper import LSCWrapper
from tune.protox.env.mqo.mqo_wrapper import MQOWrapper
from tune.protox.env.space.holon_space import HolonSpace
from tune.protox.env.space.latent_space.latent_knob_space import LatentKnobSpace
from tune.protox.env.space.latent_space.latent_query_space import LatentQuerySpace
from tune.protox.env.space.latent_space.lsc_index_space import LSCIndexSpace
from tune.protox.env.space.state import LSCMetricStateSpace, LSCStructureStateSpace
from tune.protox.env.space.state.space import StateSpace
from tune.protox.env.target_reset.target_reset_wrapper import TargetResetWrapper
from tune.protox.env.types import ProtoAction, TableAttrAccessSetsMap
from tune.protox.env.util.pg_conn import PostgresConn
from tune.protox.env.util.reward import RewardUtility
from tune.protox.env.workload import Workload


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
            s.bind(("", port))

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

            with open(f"{signal_folder}/{port}.signal", "w") as f:  # type: IO[Any]
                f.write(str(port))
                f.close()

            s.close()
            return port, f"{signal_folder}/{port}.signal"
        except OSError as e:
            port += 1
    raise IOError("No free ports to bind postgres to.")


def _modify_benchbase_config(logdir: str, port: int, hpo_params: dict[str, Any]) -> None:
    if hpo_params["benchmark_config"]["query_spec"]["oltp_workload"]:
        conf_etree = ET.parse(Path(logdir) / "benchmark.xml")
        jdbc = f"jdbc:postgresql://localhost:{port}/benchbase?preferQueryMode=extended"
        conf_etree.getroot().find("url").text = jdbc  # type: ignore

        oltp_config = hpo_params["benchbase_config"]["oltp_config"]
        if conf_etree.getroot().find("scalefactor") is not None:
            conf_etree.getroot().find("scalefactor").text = str(oltp_config["oltp_sf"])  # type: ignore
        if conf_etree.getroot().find("terminals") is not None:
            conf_etree.getroot().find("terminals").text = str(oltp_config["oltp_num_terminals"])  # type: ignore
        if conf_etree.getroot().find("works") is not None:
            works = conf_etree.getroot().find("works").find("work")  # type: ignore
            if works.find("time") is not None:  # type: ignore
                conf_etree.getroot().find("works").find("work").find("time").text = str(oltp_config["oltp_duration"])  # type: ignore
            if works.find("warmup") is not None:  # type: ignore
                conf_etree.getroot().find("works").find("work").find("warmup").text = str(oltp_config["oltp_warmup"])  # type: ignore
        conf_etree.write(Path(logdir) / "benchmark.xml")


def _gen_noise_scale(
    vae_config: dict[str, Any], hpo_params: dict[str, Any]
) -> Callable[[ProtoAction, torch.Tensor], ProtoAction]:
    def f(p: ProtoAction, n: torch.Tensor) -> ProtoAction:
        if hpo_params["scale_noise_perturb"]:
            return ProtoAction(
                torch.clamp(
                    p + n * vae_config["output_scale"], 0.0, vae_config["output_scale"]
                )
            )
        else:
            return ProtoAction(torch.clamp(p + n, 0.0, 1.0))

    return f


def _build_utilities(
    dbgym_cfg: DBGymConfig, logdir: str, pgport: int, hpo_params: dict[str, Any]
) -> Tuple[Logger, RewardUtility, PostgresConn, Workload]:
    logger = Logger(
        hpo_params["trace"],
        hpo_params["verbose"],
        Path(logdir),
        Path(logdir) / "repository",
        Path(logdir) / "tboard",
    )

    reward_utility = RewardUtility(
        target=(
            "tps"
            if hpo_params["benchmark_config"]["query_spec"]["oltp_workload"]
            else "latency"
        ),
        metric=hpo_params["reward"],
        reward_scaler=hpo_params["reward_scaler"],
        logger=logger,
    )

    # PostgresConn.start_with_changes() assumes that Redis is running
    # TODO(phw2): only start redis if we're using Boot
    make_redis_started(dbgym_cfg.root_yaml["boot_redis_port"])

    pgconn = PostgresConn(
        dbgym_cfg=dbgym_cfg,
        pgport=pgport,
        pristine_pgdata_snapshot_fpath=Path(hpo_params["pgconn_info"]["pristine_pgdata_snapshot_path"]),
        pgdata_parent_dpath=Path(hpo_params["pgconn_info"]["pgdata_parent_dpath"]),
        pgbin_path=Path(hpo_params["pgconn_info"]["pgbin_path"]),
        postgres_logs_dir=Path(logdir) / "pg_logs",
        connect_timeout=300,
        logger=logger,
    )

    workload = Workload(
        dbgym_cfg=dbgym_cfg,
        tables=hpo_params["benchmark_config"]["tables"],
        attributes=hpo_params["benchmark_config"]["attributes"],
        query_spec=hpo_params["benchmark_config"]["query_spec"],
        workload_path=Path(hpo_params["workload_path"]),
        pid=None,
        workload_timeout=hpo_params["workload_timeout"],
        workload_timeout_penalty=hpo_params["workload_timeout_penalty"],
        logger=logger,
    )

    return logger, reward_utility, pgconn, workload


def _build_actions(
    dbgym_cfg: DBGymConfig, seed: int, hpo_params: dict[str, Any], workload: Workload, logger: Logger
) -> Tuple[HolonSpace, LSC]:
    sysknobs = LatentKnobSpace(
        logger=logger,
        tables=hpo_params["benchmark_config"]["tables"],
        knobs=hpo_params["system_knobs"],
        quantize=True,
        quantize_factor=hpo_params["default_quantization_factor"],
        seed=seed,
        table_level_knobs=hpo_params["benchmark_config"]["table_level_knobs"],
        latent=True,
    )

    with open_and_save(dbgym_cfg, Path(hpo_params["embedder_path"]) / "config") as f:
        vae_config = json.load(f)

        assert vae_config["mean_output_act"] == "sigmoid"
        index_output_transform = (
            lambda x: torch.nn.Sigmoid()(x) * vae_config["output_scale"]
        )
        index_noise_scale = _gen_noise_scale(vae_config, hpo_params)

        max_attrs, max_cat_features = fetch_vae_parameters_from_workload(
            workload, len(hpo_params["benchmark_config"]["tables"])
        )
        vae = create_vae_model(vae_config, max_attrs, max_cat_features)
        embedder_fpath = Path(hpo_params["embedder_path"]) / "embedder.pth"
        save_file(dbgym_cfg, embedder_fpath)
        vae.load_state_dict(torch.load(embedder_fpath))

    lsc = LSC(
        horizon=hpo_params["horizon"],
        lsc_parameters=hpo_params["lsc"],
        vae_config=vae_config,
        logger=logger,
    )

    idxspace = LSCIndexSpace(
        tables=hpo_params["benchmark_config"]["tables"],
        max_num_columns=hpo_params["benchmark_config"]["max_num_columns"],
        max_indexable_attributes=workload.max_indexable(),
        seed=seed,
        # TODO(wz2): We should theoretically pull this from the DBMS.
        rel_metadata=hpo_params["benchmark_config"]["attributes"],
        attributes_overwrite=workload.column_usages(),
        tbl_include_subsets=TableAttrAccessSetsMap(workload.tbl_include_subsets),
        index_space_aux_type=hpo_params["benchmark_config"]["index_space_aux_type"],
        index_space_aux_include=hpo_params["benchmark_config"][
            "index_space_aux_include"
        ],
        deterministic_policy=True,
        vae=vae,
        latent_dim=vae_config["latent_dim"],
        index_output_transform=index_output_transform,
        index_noise_scale=index_noise_scale,
        logger=logger,
        lsc=lsc,
    )

    qspace = LatentQuerySpace(
        tables=hpo_params["benchmark_config"]["tables"],
        quantize=True,
        quantize_factor=hpo_params["default_quantization_factor"],
        seed=seed,
        per_query_knobs_gen=hpo_params["benchmark_config"]["per_query_knob_gen"],
        per_query_parallel=(
            {}
            if not hpo_params["benchmark_config"]["per_query_select_parallel"]
            else workload.query_aliases
        ),
        per_query_scans=(
            {}
            if not hpo_params["benchmark_config"]["per_query_scan_method"]
            else workload.query_aliases
        ),
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


def _build_obs_space(
    dbgym_cfg: DBGymConfig, action_space: HolonSpace, lsc: LSC, hpo_params: dict[str, Any], seed: int
) -> StateSpace:
    if hpo_params["metric_state"] == "metric":
        return LSCMetricStateSpace(
            dbgym_cfg=dbgym_cfg,
            lsc=lsc,
            tables=hpo_params["benchmark_config"]["tables"],
            seed=seed,
        )
    elif hpo_params["metric_state"] == "structure":
        return LSCStructureStateSpace(
            lsc=lsc,
            action_space=action_space,
            normalize=False,
            seed=seed,
        )
    elif hpo_params["metric_state"] == "structure_normalize":
        return LSCStructureStateSpace(
            lsc=lsc,
            action_space=action_space,
            normalize=True,
            seed=seed,
        )
    else:
        ms = hpo_params["metric_state"]
        raise ValueError(f"Unsupported state representation {ms}")


def _build_env(
    dbgym_cfg: DBGymConfig,
    hpo_params: dict[str, Any],
    pgconn: PostgresConn,
    obs_space: StateSpace,
    holon_space: HolonSpace,
    lsc: LSC,
    workload: Workload,
    reward_utility: RewardUtility,
    logger: Logger,
) -> Tuple[TargetResetWrapper, AgentEnv]:

    env = gym.make(
        "Postgres-v0",
        dbgym_cfg=dbgym_cfg,
        observation_space=obs_space,
        action_space=holon_space,
        workload=workload,
        horizon=hpo_params["horizon"],
        reward_utility=reward_utility,
        pgconn=pgconn,
        pqt=hpo_params["query_timeout"],
        benchbase_config=hpo_params["benchbase_config"],
        logger=logger,
        replay=False,
    )

    # Check whether to create the MQO wrapper.
    if not hpo_params["benchmark_config"]["query_spec"]["oltp_workload"]:
        if (
            hpo_params["workload_eval_mode"] != "pq"
            or hpo_params["workload_eval_inverse"]
            or hpo_params["workload_eval_reset"]
        ):
            env = MQOWrapper(
                workload_eval_mode=hpo_params["workload_eval_mode"],
                workload_eval_inverse=hpo_params["workload_eval_inverse"],
                workload_eval_reset=hpo_params["workload_eval_reset"],
                benchbase_config=hpo_params["benchbase_config"],
                pqt=hpo_params["query_timeout"],
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
        maximize_state=hpo_params["maximize_state"],
        reward_utility=reward_utility,
        start_reset=False,
        logger=logger,
    )

    env = FlattenObservation(env)
    if hpo_params["normalize_state"]:
        env = NormalizeObservation(env)

    if hpo_params["normalize_reward"]:
        env = NormalizeReward(env, gamma=hpo_params["gamma"])

    # Wrap the AgentEnv to have null checking.
    env = AgentEnv(env)
    return target_reset, env


def _build_agent(
    seed: int,
    hpo_params: dict[str, Any],
    obs_space: StateSpace,
    action_space: HolonSpace,
    logger: Logger,
) -> Wolp:
    action_dim = noise_action_dim = action_space.latent_dim()
    critic_action_dim = action_space.critic_dim()

    actor = Actor(
        observation_space=obs_space,
        action_space=action_space,
        net_arch=[int(l) for l in hpo_params["pi_arch"].split(",")],
        features_dim=gym.spaces.utils.flatdim(obs_space),
        activation_fn=_parse_activation_fn(hpo_params["activation_fn"]),
        weight_init=hpo_params["weight_init"],
        bias_zero=hpo_params["bias_zero"],
        squash_output=False,
        action_dim=action_dim,
        policy_weight_adjustment=hpo_params["policy_weight_adjustment"],
    )

    actor_target = Actor(
        observation_space=obs_space,
        action_space=action_space,
        net_arch=[int(l) for l in hpo_params["pi_arch"].split(",")],
        features_dim=gym.spaces.utils.flatdim(obs_space),
        activation_fn=_parse_activation_fn(hpo_params["activation_fn"]),
        weight_init=hpo_params["weight_init"],
        bias_zero=hpo_params["bias_zero"],
        squash_output=False,
        action_dim=action_dim,
        policy_weight_adjustment=hpo_params["policy_weight_adjustment"],
    )

    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=hpo_params["learning_rate"]
    )

    critic = ContinuousCritic(
        observation_space=obs_space,
        action_space=action_space,
        net_arch=[int(l) for l in hpo_params["qf_arch"].split(",")],
        features_dim=gym.spaces.utils.flatdim(obs_space),
        activation_fn=_parse_activation_fn(hpo_params["activation_fn"]),
        weight_init=hpo_params["weight_init"],
        bias_zero=hpo_params["bias_zero"],
        n_critics=2,
        action_dim=critic_action_dim,
    )

    critic_target = ContinuousCritic(
        observation_space=obs_space,
        action_space=action_space,
        net_arch=[int(l) for l in hpo_params["qf_arch"].split(",")],
        features_dim=gym.spaces.utils.flatdim(obs_space),
        activation_fn=_parse_activation_fn(hpo_params["activation_fn"]),
        weight_init=hpo_params["weight_init"],
        bias_zero=hpo_params["bias_zero"],
        n_critics=2,
        action_dim=critic_action_dim,
    )

    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=hpo_params["learning_rate"] * hpo_params["critic_lr_scale"],
    )

    policy = WolpPolicy(
        observation_space=obs_space,
        action_space=action_space,
        actor=actor,
        actor_target=actor_target,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_target=critic_target,
        critic_optimizer=critic_optimizer,
        grad_clip=hpo_params["grad_clip"],
        policy_l2_reg=hpo_params["policy_l2_reg"],
        tau=hpo_params["tau"],
        gamma=hpo_params["gamma"],
        logger=logger,
    )

    # Setup the noise policy.
    noise_params = hpo_params["noise_parameters"]
    means = np.zeros((noise_action_dim,), dtype=np.float32)
    stddevs = np.full(
        (noise_action_dim,), noise_params["noise_sigma"], dtype=np.float32
    )
    action_noise_type = parse_noise_type(noise_params["noise_type"])
    action_noise = None if not action_noise_type else action_noise_type(means, stddevs)

    target_noise = hpo_params["target_noise"]
    means = np.zeros(
        (
            hpo_params["batch_size"],
            noise_action_dim,
        ),
        dtype=np.float32,
    )
    stddevs = np.full(
        (
            hpo_params["batch_size"],
            noise_action_dim,
        ),
        target_noise["target_policy_noise"],
        dtype=np.float32,
    )
    target_action_noise = parse_noise_type("normal")
    assert target_action_noise
    clamp_noise = ClampNoise(
        target_action_noise(means, stddevs), target_noise["target_noise_clip"]
    )

    return Wolp(
        policy=policy,
        replay_buffer=ReplayBuffer(
            buffer_size=hpo_params["buffer_size"],
            obs_shape=[gym.spaces.utils.flatdim(obs_space)],
            action_dim=critic_action_dim,
        ),
        learning_starts=hpo_params["learning_starts"],
        batch_size=hpo_params["batch_size"],
        train_freq=(hpo_params["train_freq_frequency"], hpo_params["train_freq_unit"]),
        gradient_steps=hpo_params["gradient_steps"],
        action_noise=action_noise,
        target_action_noise=clamp_noise,
        seed=seed,
        neighbor_parameters=hpo_params["neighbor_parameters"],
    )


def build_trial(
    dbgym_cfg: DBGymConfig, seed: int, logdir: str, hpo_params: dict[str, Any]
) -> Tuple[Logger, TargetResetWrapper, AgentEnv, Wolp, str]:
    # The massive trial builder.

    port, signal = _get_signal(hpo_params["pgconn_info"]["pgbin_path"])
    _modify_benchbase_config(logdir, port, hpo_params)

    logger, reward_utility, pgconn, workload = _build_utilities(dbgym_cfg, logdir, port, hpo_params)
    holon_space, lsc = _build_actions(dbgym_cfg, seed, hpo_params, workload, logger)
    obs_space = _build_obs_space(dbgym_cfg, holon_space, lsc, hpo_params, seed)
    target_reset, env = _build_env(
        dbgym_cfg,
        hpo_params,
        pgconn,
        obs_space,
        holon_space,
        lsc,
        workload,
        reward_utility,
        logger,
    )

    agent = _build_agent(seed, hpo_params, obs_space, holon_space, logger)
    return logger, target_reset, env, agent, signal
