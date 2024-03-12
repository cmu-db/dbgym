import logging
import time
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, NormalizeReward, FlattenObservation
import pickle

from tune.protox.env.spec import Spec
from tune.protox.env.reward import RewardUtility
from tune.protox.utils.target_state_reset import TargetStateResetWrapper
from tune.protox.utils.mythril_vec_env_wrapper import MythrilVecEnvWrapper
from tune.protox.agent.vec_env import VecCheckNan
from tune.protox.utils.dotdict import DotDict
from tune.protox.agent.wolp.config import setup_wolp_agent


class TimeoutChecker(object):
    running = False
    limit = False
    remain = 0
    start = 0

    def __init__(self, duration):
        self.limit = (duration * 3600) > 0
        self.remain = int(duration * 3600)

    def resume(self):
        self.start = time.time()
        self.running = True

    def pause(self):
        if self.limit and self.running:
            self.remain -= int(time.time() - self.start)
        self.running = False

    def __call__(self):
        if not self.limit:
            return False

        if self.remain <= 0:
            return True

        if self.running:
            return int(time.time() - self.start) >= self.remain

        return False


class TuneTrial(object):
    def setup(self, dbgym_cfg, is_oltp, args, timeout_checker):
        torch.set_default_dtype(torch.float32)
        self.args = DotDict(args)

        # Reset variables.
        self.accumulated_reward = 0
        self.episode_step = 0
        self.global_policy_loss = 0
        self.global_critic_loss = 0
        self.it = 0

        # Setup the seed
        seed = self.args.seed if self.args.seed != -1 else np.random.randint(np.iinfo(np.int32).max)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load spec and create environment.
        self.spec = Spec(dbgym_cfg, self.args.agent, seed, embedding_path=self.args.embedding_path, protox_config_path=self.args.protox_config_path, benchbase_config_path=self.args.benchbase_config_path, benchmark_config_path=self.args.benchmark_config_path, workload_path=self.args.workload_path, horizon=self.args.horizon, workload_timeout=self.args.workload_timeout)
        self.logger = self.spec.logger
        logging.info("%s", args)
        logging.info(f"Seed: {seed}")

        # Create the reward utility.
        self.reward_utility = RewardUtility(is_oltp, self.args.reward, self.spec.reward_scaler)

        # Create the environment.
        self.base_env = self.env = gym.make("Postgres-v0",
            spec=self.spec,
            horizon=self.args.horizon,
            query_timeout=self.args.query_timeout,
            reward_utility=self.reward_utility,
            logger=self.logger)

        # Attach a target state reset wrapper around it.
        mko = self.spec.maximize_knobs_only if hasattr(self.spec, "maximize_knobs_only") else False
        sr = self.spec.start_reset if hasattr(self.spec, "start_reset") else False
        self.target_state_wrapper = self.env = TargetStateResetWrapper(self.env, self.spec.maximize_state, self.reward_utility, maximize_knobs_only=mko, start_reset=sr)

        # Now attach a flatten observation wrapper.
        self.env = FlattenObservation(self.env)
        self.normalize_env = None
        self.normalize_reward = None
        if self.spec.normalize_state:
            # Normalize state if we are asked to.
            self.normalize_env = self.env = NormalizeObservation(self.env)
        if self.spec.normalize_reward:
            # Normalize reward if we are asked to.
            self.normalize_reward = self.env = NormalizeReward(self.env, gamma=self.spec.gamma)
        self.vec_env = self.env = MythrilVecEnvWrapper([lambda: self.env])
        self.env = VecCheckNan(self.env, raise_exception=True, warn_once=False)

        # Setup the agent.
        self.agent = setup_agent(dbgym_cfg, self.env, self.spec, seed, self.args.agent, self.args.agent_params_path)
        self.agent.set_logger(self.logger)
        self.agent.set_timeout_checker(timeout_checker)
        self.env_init = False

        if "load" in self.args:
            # Un-pickle.
            self.unpickle()


    def step(self):
        if not self.env_init:
            # Create the baseline repository data.
            self.state, infos = self.env.reset()
            info = infos[0]
            baseline_reward = info["baseline_reward"]
            self.baseline_metric = info["baseline_metric"]
            logging.info(f"Baseilne Metric: {self.baseline_metric}. Baseline Reward: {baseline_reward}")
            self.env_init = True

        episode = self.agent._episode_num
        it = self.agent.num_timesteps
        logging.info(f"Starting episode: {episode+1}, iteration: {it+1}")

        self.agent.learn(
            total_timesteps=1,
            log_interval=None,
            reset_num_timesteps=False,
            progress_bar=False)

        self.logger.advance()

        return {
            "Episode": episode,
            "Timesteps": it,
            "Best Metric": self.target_state_wrapper.real_best_metric,
            "Best Seen Metric": self.target_state_wrapper.best_metric,
        }

    def cleanup(self):
        self.logger.flush()
        if "dump" in self.args and self.args.dump == 1:
            # Pickle before we close.
            self.pickle()
        self.env.close()

    def unpickle(self):
        with open(self.spec.dump_path, "rb") as f:
            data = pickle.load(f)

        np.random.set_state(data["np_seed"])
        torch.set_rng_state(data["th_seed"])
        self.spec.load_state(data["spec"])
        self.reward_utility.load_state(data["reward_utility"])

        if self.normalize_reward:
            # Load reward.
            self.normalize_reward.return_rms  = data["normalize_reward"].return_rms
            self.normalize_reward.returns = data["normalize_reward"].returns
            self.normalize_reward.gamma = data["normalize_reward"].gamma
            self.normalize_reward.epsilon = data["normalize_reward"].epsilon

        if self.normalize_env:
            # Load normalize env.
            self.normalize_env.obs_rms = data["normalize_env"]["obs_rms"]
            self.normalize_env.epsilon = data["normalize_env"]["epsilon"]

        self.vec_env.load_state(data["vec_env"])
        self.target_state_wrapper.load_state(data["target_state"])
        self.base_env.load_state(data["base_env"])
        self.agent.load_state(data["agent"])
        self.base_env.restore_pristine_snapshot()
        self.env.reset(options={"load": True})
        self.env_init = True

    def pickle(self):
        pickle_data = {
            "np_seed": np.random.get_state(),
            "th_seed": torch.get_rng_state(),
            "spec": self.spec.save_state(),
            "reward_utility": self.reward_utility,
            "normalize_reward": self.normalize_reward,
            "normalize_env": { "obs_rms": self.normalize_env.obs_rms, "epsilon": self.normalize_env.epsilon },
            "vec_env": self.vec_env.save_state(),
            "target_state": self.target_state_wrapper.save_state(),
            "base_env": self.base_env.save_state(),
            "agent": self.agent.save_state(),
        }

        with open(self.spec.dump_path, "wb") as f:
            pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def setup_agent(dbgym_cfg, env, spec, seed, agent_type, agent_config):
    if agent_type == "wolp":
        return setup_wolp_agent(dbgym_cfg, env, spec, seed, agent_config)
    else:
        assert False, f"Unsupported agent {agent_type}"