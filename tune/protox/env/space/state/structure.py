from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union, cast

import gymnasium as gym
import numpy as np
import psycopg
import torch as th
from gymnasium import spaces

from tune.protox.env.space.holon_space import HolonSpace
from tune.protox.env.space.latent_space import (
    LatentIndexSpace,
    LatentKnobSpace,
    LatentQuerySpace,
)
from tune.protox.env.space.primitive.index import IndexAction
from tune.protox.env.space.state.space import StateSpace
from tune.protox.env.space.utils import check_subspace
from tune.protox.env.types import IndexSpaceRawSample, KnobSpaceAction, QuerySpaceAction
from util.workspace import DBGymConfig


class StructureStateSpace(StateSpace, spaces.Dict):
    def __init__(
        self,
        action_space: HolonSpace,
        spaces: Mapping[str, spaces.Space[Any]],
        normalize: bool,
        seed: int,
    ) -> None:
        self.action_space = action_space
        self.normalize = normalize

        if self.normalize:
            self.internal_spaces: dict[str, gym.spaces.Space[Any]] = {
                k: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(s.critic_dim(),))
                for k, s in action_space.get_spaces()
            }
        else:
            self.internal_spaces = {
                k: (
                    gym.spaces.Box(low=-np.inf, high=np.inf, shape=(s.critic_dim(),))
                    if s.uses_embed()
                    else s
                )
                for k, s in action_space.get_spaces()
            }

        self.internal_spaces.update(spaces)
        super().__init__(self.internal_spaces, seed)

    def require_metrics(self) -> bool:
        return False

    def check_benchbase(
        self, dbgym_cfg: DBGymConfig, results_dpath: Union[str, Path]
    ) -> bool:
        # We don't use benchbase metrics anyways.
        return True

    def construct_offline(
        self, connection: psycopg.Connection[Any], data: Any, prev_state_container: Any
    ) -> dict[str, Any]:
        assert isinstance(self.action_space, HolonSpace)
        splits = self.action_space.split_action(prev_state_container)
        knob_states = [v[1] for v in splits if isinstance(v[0], LatentKnobSpace)]
        knob_state = cast(
            Optional[KnobSpaceAction], None if len(knob_states) == 0 else knob_states[0]
        )

        ql_states = [v[1] for v in splits if isinstance(v[0], LatentQuerySpace)]
        ql_state = cast(
            Optional[QuerySpaceAction], None if len(ql_states) == 0 else ql_states[0]
        )

        if knob_state is not None:
            knobs = self.action_space.get_knob_space()
            assert isinstance(knobs, LatentKnobSpace)
            assert check_subspace(knobs, knob_state)

            if self.normalize:
                knob_state = np.array(
                    knobs.to_latent([knob_state]),
                    dtype=np.float32,
                )[0]
            assert self.internal_spaces["knobs"].contains(knob_state)

        if ql_state is not None:
            query = self.action_space.get_query_space()
            assert isinstance(query, LatentQuerySpace)
            assert check_subspace(query, ql_state)

            if self.normalize:
                query_state = np.array(
                    query.to_latent([ql_state]),
                    dtype=np.float32,
                )[0]
            else:
                query_state = ql_state

        # Handle indexes.
        indexes_ = [v[1] for v in splits if isinstance(v[0], LatentIndexSpace)]
        indexes = cast(list[IndexAction], None if len(indexes_) == 0 else indexes_[0])
        index_state = None

        if indexes is not None:
            index_space = self.action_space.get_index_space()
            # TODO(wz2): Incorporate existing stock-config indexes into state and then make the guard len(indexes) > 0 instead of len(env_acts) > 0
            env_acts = [v.raw_repr for v in indexes if v.raw_repr]
            if len(env_acts) > 0:
                with th.no_grad():
                    latents = index_space.to_latent(env_acts).numpy()
                    latents = latents.sum(axis=0)
                    latents /= len(indexes)
                    index_state = latents.flatten().astype(np.float32)
            else:
                index_state = np.zeros(index_space.critic_dim(), dtype=np.float32)

        state: dict[str, Any] = {}
        if knob_state is not None:
            state["knobs"] = knob_state
        if query_state is not None:
            state["query"] = query_state
        if index_state is not None:
            state["index"] = index_state
        return state

    def construct_online(self, connection: psycopg.Connection[Any]) -> dict[str, Any]:
        raise NotImplementedError()

    def state_delta(
        self, initial: dict[str, Any], final: dict[str, Any]
    ) -> dict[str, Any]:
        raise NotImplementedError()

    def merge_deltas(self, merge_data: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError()
