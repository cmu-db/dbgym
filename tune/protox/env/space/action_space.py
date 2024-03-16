import itertools
import time

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


class ActionSpace(spaces.Tuple):
    dims = None
    knob_space_ind = None
    index_space_ind = None
    lsc_embed = True

    def save_state(self):
        saves = []
        for s in self.spaces:
            saves.append(s.save_state())

        return {
            "saves": saves,
        }

    def load_state(self, data):
        for i, s in enumerate(data["saves"]):
            self.spaces[i].load_state(s)

    def __init__(self, knob_space, index_space, seed, lsc_embed=True):
        spaces = []
        knob_space_ind = None
        index_space_ind = None
        if knob_space is not None:
            knob_space_ind = len(spaces)
            spaces.append(knob_space)
        if index_space is not None:
            index_space_ind = len(spaces)
            spaces.append(index_space)
        super().__init__(spaces, seed=seed)
        self.knob_space_ind = knob_space_ind
        self.index_space_ind = index_space_ind
        self.lsc_embed = lsc_embed

        # Check that we actually have a latent dimension.
        self.contains_latent = False
        for s in self.spaces:
            if s.latent:
                self.contains_latent = True

        raw_dims = [
            (
                gym.spaces.utils.flatdim(space)
                if space.get_latent_dim() == 0
                else space.get_latent_dim()
            )
            for space in self.spaces
        ]
        self.raw_dims = np.cumsum(raw_dims)
        self.space_dims = np.cumsum([gym.spaces.utils.flatdim(s) for s in self.spaces])

    def get_current_lsc(self):
        idxs = self.get_index_space()
        if idxs is None or idxs.lsc is None or (not self.lsc_embed):
            return np.array([-1.0], dtype=np.float32)
        return idxs.lsc.current_scale()

    def adjust_action_lsc(self, act, lsc):
        idxs = self.get_index_space()
        if idxs is None or idxs.lsc is None or (not self.lsc_embed):
            return act

        assert len(self.raw_dims) <= 2
        if len(self.raw_dims) == 2:
            a0 = act[:, : self.raw_dims[0]]
            a1 = act[:, self.raw_dims[0] :] + idxs.lsc.inverse_scale(lsc)
            return torch.concat([a0, a1], dim=1)
        else:
            return act + idxs.lsc.inverse_scale(lsc)

    def get_latent_dim(self):
        return self.raw_dims[-1]

    def get_critic_dim(self):
        return sum([space.get_critic_dim() for space in self.spaces])

    def get_knob_space(self):
        if self.knob_space_ind is None:
            return None
        return self.spaces[self.knob_space_ind]

    def get_index_space(self):
        if self.index_space_ind is None:
            return None
        return self.spaces[self.index_space_ind]

    def get_state(self, env):
        return tuple([space.get_state(env) for space in self.spaces])

    def get_query_level_knobs(self, action):
        if self.knob_space_ind is None:
            return {}

        return self.spaces[self.knob_space_ind].get_query_level_knobs(
            action[self.knob_space_ind]
        )

    def reset(self, **kwargs):
        # Reset our understanding of the space.
        _ = [space.reset(**kwargs) for space in self.spaces]

    def advance(self, action, **kwargs):
        # Step our understanding of the space of valid actions.
        _ = [space.advance(action[i], **kwargs) for i, space in enumerate(self.spaces)]

    def decode(self, protos):
        components = []
        for i, s in enumerate(self.spaces):
            start = self.raw_dims[i - 1] if i > 0 else 0
            end = self.raw_dims[i]
            if len(protos.shape) == 2:
                components.append(s._decode(protos[:, start:end]))
            else:
                components.append(s._decode(protos[start:end].reshape(1, -1)))
        return torch.concat(components, dim=1)

    def process_network_output(self, proto):
        for i, s in enumerate(self.spaces):
            start = self.raw_dims[i - 1] if i > 0 else 0
            end = self.raw_dims[i]
            if len(proto.shape) == 2:
                proto[:, start:end] = s._process_network_output(proto[:, start:end])
            else:
                proto[start:end] = s._process_network_output(proto[start:end])
        return proto

    def perturb_noise(self, proto, noise):
        for i, s in enumerate(self.spaces):
            start = self.raw_dims[i - 1] if i > 0 else 0
            end = self.raw_dims[i]

            # Extract the noise component.
            if len(noise.shape) == 2:
                noise_cmp = noise[:, start:end]
            else:
                noise_cmp = noise[start:end]

            if len(proto.shape) == 2:
                proto[:, start:end] = s._perturb_noise(proto[:, start:end], noise_cmp)
            else:
                proto[start:end] = s._perturb_noise(proto[:, start:end], noise_cmp)
        return proto

    def actor_smear_action(self, act, neighbor_parameters={}, random=False):
        # This function returns three elements:
        # (1) Environment action.
        # (2) Action embedding representation.
        # (3) Array dimensions to demarcate act in case act is batched.
        #
        # This is because we want the critic to operate on the action embedding
        # and not on the environment action. The environment action is simply
        # action that we end up wanting to apply to the environment.
        protos = act.cpu()

        if self.contains_latent:
            start_time = time.time()
            # Always attempt to decode because you never know what is there.
            protos = self.decode(protos)

        env_acts = []
        emb_acts = []
        ndims = []

        search_times = []
        product_times = []

        for proto in protos:
            # Figure out the neighbors for each subspace.
            envs_neighbors = []
            start_time = time.time()
            for i, s in enumerate(self.spaces):
                # Note that subproto is the "action embedding" from the context of
                # the embedded actor-critic like architecutre.
                subproto = proto[
                    (self.space_dims[i - 1] if i > 0 else 0) : self.space_dims[i]
                ]
                envs = s.search_embedding_neighborhood(subproto, neighbor_parameters)

                if random:
                    # Draw a random action.
                    rand_idx = np.random.randint(0, high=len(envs))
                    envs = [envs[rand_idx]]

                envs_neighbors.append(envs)
            search_times.append(time.time() - start_time)
            start_time = time.time()

            # Cartesian product itself is naturally in the joint space.
            envs_neighbors = [l for l in itertools.product(*envs_neighbors)]
            product_times.append(time.time() - start_time)
            embed_neighbors = self.actor_action_embedding(envs_neighbors)

            assert len(envs_neighbors) > 0
            env_acts.extend(envs_neighbors)
            emb_acts.extend(embed_neighbors)
            ndims.append(len(envs_neighbors))

        return env_acts, np.array(emb_acts), np.array(ndims)

    def actor_action_embedding(self, env_act):
        # This function returns the current action embedding associated
        # with the environment action. This provides the inverse from
        # the environment action to the action embedding space.
        if not isinstance(env_act, list):
            env_act = [env_act]
        embed_cmps = [
            np.array(s._env_to_embedding([a[i] for a in env_act]))
            for i, s in enumerate(self.spaces)
        ]
        return np.concatenate(embed_cmps, axis=1)

    def random_embed_action(self, num_action=1):
        subspaces = [space._random_embed_action(num_action) for space in self.spaces]
        return np.concatenate(subspaces, axis=1)

    def sample(self):
        assert False

    def generate_action_plan(self, action, **kwargs):
        outputs = [
            space.generate_action_plan(action[i], **kwargs)
            for i, space in enumerate(self.spaces)
        ]
        cc = list(itertools.chain(*[o[0] for o in outputs]))
        sql_commands = list(itertools.chain(*[o[1] for o in outputs]))
        return cc, sql_commands

    def generate_plan_from_config(self, config, **kwargs):
        kwargs["reset"] = True
        outputs = [
            space.generate_delta_action_plan(config[i], **kwargs)
            for i, space in enumerate(self.spaces)
        ]
        config_changes = list(itertools.chain(*[o[0] for o in outputs]))
        sql_commands = list(itertools.chain(*[o[1] for o in outputs]))
        return config_changes, sql_commands
