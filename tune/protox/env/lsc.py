import numpy as np
import torch
import logging


class LSC(object):
    # Horizon for an episode.
    horion = 0
    num_steps = 0
    num_episodes = 0

    lsc_shift = None
    lsc_shift_increment = None
    lsc_shift_max = None
    lsc_shift_after = 0
    lsc_shift_schedule_eps_freq: int = 1

    vae_configuration: dict = {}

    def __init__(self, spec, horizon: int, vae_config: dict):
        assert spec.index_vae_metadata["index_vae"]
        self.frozen = False
        self.horizon = horizon
        self.num_steps = 0
        self.num_episodes = 0
        self.vae_configuration = vae_config

        lsc_splits = spec.lsc_parameters["lsc_shift_initial"].split(",")
        lsc_increments = spec.lsc_parameters["lsc_shift_increment"].split(",")
        lsc_max = spec.lsc_parameters["lsc_shift_max"].split(",")
        if len(lsc_splits) == 1:
            lsc_splits = [float(lsc_splits[0])] * horizon
        else:
            assert len(lsc_splits) == horizon
            lsc_splits = [float(f) for f in lsc_splits]

        if len(lsc_increments) == 1:
            lsc_increments = [float(lsc_increments[0])] * horizon
        else:
            assert len(lsc_increments) == horizon
            lsc_increments = [float(f) for f in lsc_increments]

        if len(lsc_max) == 1:
            lsc_max = [float(lsc_max[0])] * horizon
        else:
            assert len(lsc_max) == horizon
            lsc_max = [float(f) for f in lsc_max]

        self.lsc_shift_schedule_eps_freq = spec.lsc_parameters["lsc_shift_schedule_eps_freq"]
        self.lsc_shift = np.array(lsc_splits)
        self.lsc_shift_increment = np.array(lsc_increments)
        self.lsc_shift_max = np.array(lsc_max)
        self.lsc_shift_after = spec.lsc_parameters["lsc_shift_after"]
        logging.info("LSC Shift: %s", self.lsc_shift)
        logging.info("LSC Shift Increment: %s", self.lsc_shift_increment)
        logging.info("LSC Shift Max: %s", self.lsc_shift_max)

    def apply_current_bias(self, action):
        assert action.shape[-1] == self.vae_configuration["latent_dim"]

        # Get the LSC shift associated with the current episode.
        lsc_shift = self.lsc_shift[(self.num_steps % self.horizon)]
        lsc_shift = lsc_shift * self.vae_configuration["output_scale"]
        return action + lsc_shift

    def current_bias(self):
        # Get the LSC shift associated with the current episode.
        lsc_shift = self.lsc_shift[(self.num_steps % self.horizon)]
        lsc_shift = lsc_shift * self.vae_configuration["output_scale"]
        return lsc_shift

    def current_scale(self):
        lsc_shift = self.lsc_shift[(self.num_steps % self.horizon)]
        lsc_max = self.lsc_shift_max[(self.num_steps % self.horizon)]
        rel = lsc_shift / lsc_max
        return np.array([(rel * 2.) - 1], dtype=np.float32)

    def inverse_scale(self, value):
        lsc_max = self.lsc_shift_max[0]
        lsc_shift = ((value + 1) / 2.) * lsc_max
        return lsc_shift * self.vae_configuration["output_scale"]

    def advance(self):
        if self.frozen:
            return

        self.num_steps += 1

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def reset(self):
        if self.frozen:
            return

        # Advance the episode count.
        self.num_episodes += 1
        if (self.num_episodes <= self.lsc_shift_after) or ((self.num_episodes - self.lsc_shift_after) % self.lsc_shift_schedule_eps_freq != 0):
            # Reset the number of steps we've taken.
            self.num_steps = 0
        else:
            # Get how many steps to make the update on.
            bound = self.horizon
            self.num_steps = 0

            # Now try to perform the LSC shifts.
            # Increment the current bias with the increment.
            self.lsc_shift[:bound] += self.lsc_shift_increment[:bound]
            self.lsc_shift = self.lsc_shift % self.lsc_shift_max
            logging.info("LSC Bias Update: %s", self.lsc_shift)
