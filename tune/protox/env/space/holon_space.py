import copy
import itertools
from typing import Any, Iterable, List, Optional, Tuple, Union, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from psycopg import Connection

from tune.protox.env.logger import ArtifactManager, time_record
from tune.protox.env.space.latent_space import (
    LatentIndexSpace,
    LatentKnobSpace,
    LatentQuerySpace,
    LSCIndexSpace,
)
from tune.protox.env.space.utils import check_subspace
from tune.protox.env.types import (
    DEFAULT_NEIGHBOR_PARAMETERS,
    HolonAction,
    HolonStateContainer,
    HolonSubAction,
    IndexSpaceRawSample,
    KnobSpaceAction,
    NeighborParameters,
    ProtoAction,
    QuerySpaceAction,
    QuerySpaceKnobAction,
    QueryType,
)

HolonSubSpace = Union[LatentKnobSpace, LatentIndexSpace, LatentQuerySpace]


class HolonSpace(spaces.Tuple):
    def _latent_assert_check(
        self,
        carprod_neighbors: list[HolonAction],
        carprod_embeds: torch.Tensor,
        first_drift: int,
    ) -> None:
        zero = self.to_latent([carprod_neighbors[0]])[0]
        last = self.to_latent([carprod_neighbors[-1]])[0]
        first_d = self.to_latent([carprod_neighbors[first_drift]])[0]

        def eq_fn(x: torch.Tensor, y: torch.Tensor) -> bool:
            return bool(torch.isclose(x, y, atol=0.001).all().item())

        assert eq_fn(zero, carprod_embeds[0]), print(zero, carprod_embeds[0])
        assert eq_fn(last, carprod_embeds[-1]), print(last, carprod_embeds[-1])
        assert eq_fn(first_d, carprod_embeds[first_drift]), print(
            first_d, carprod_embeds[first_drift]
        )

        if self.logger:
            self.logger.get_logger(__name__).debug("Neighborhood Check passed.")

    def __init__(
        self,
        knob_space: LatentKnobSpace,
        index_space: LatentIndexSpace,
        query_space: LatentQuerySpace,
        seed: int,
        logger: Optional[ArtifactManager],
    ):
        spaces: Iterable[gym.spaces.Space[Any]] = [knob_space, index_space, query_space]
        super().__init__(spaces, seed=seed)

        raw_dims = [
            (
                gym.spaces.utils.flatdim(space)
                if space.latent_dim() == 0
                else space.latent_dim()
            )
            for space in self.spaces
            if hasattr(space, "latent_dim")
        ]
        assert len(raw_dims) == 3
        self.raw_dims: list[int] = np.cumsum(raw_dims)
        self.space_dims: Optional[list[int]] = None
        self.logger = logger

    def get_spaces(self) -> list[tuple[str, HolonSubSpace]]:
        r = cast(
            list[tuple[str, HolonSubSpace]],
            [(s.name, s) for s in self.spaces if hasattr(s, "name")],
        )
        assert len(r) == 3
        return r

    def null_action(self, sc: HolonStateContainer) -> HolonAction:
        assert isinstance(self.spaces[1], LatentIndexSpace)
        null_index = self.spaces[1].null_action()
        return HolonAction(
            (cast(KnobSpaceAction, sc[0]), null_index, cast(QuerySpaceAction, sc[2]))
        )

    def split_action(
        self, action: HolonAction
    ) -> list[tuple[HolonSubSpace, HolonSubAction]]:
        return [
            (cast(LatentKnobSpace, self.spaces[0]), action[0]),
            (cast(LatentIndexSpace, self.spaces[1]), action[1]),
            (cast(LatentQuerySpace, self.spaces[2]), action[2]),
        ]

    def extract_query(
        self, action: Union[HolonAction, HolonStateContainer]
    ) -> QuerySpaceKnobAction:
        for i, s in enumerate(self.spaces):
            if isinstance(s, LatentQuerySpace):
                q_act = action[i]
                return s.extract_query(cast(QuerySpaceAction, q_act))
        raise ValueError("Missing query space in configuration")

    def extract_knobs(self, action: HolonAction) -> Optional[KnobSpaceAction]:
        assert isinstance(self.spaces[0], LatentKnobSpace)
        return action[0]

    def replace_query(
        self, action: HolonAction, query: QuerySpaceKnobAction
    ) -> HolonAction:
        action = copy.deepcopy(action)
        for i, s in enumerate(self.spaces):
            if isinstance(s, LatentQuerySpace):
                qknobs = s.replace_query(query)
                assert check_subspace(s, qknobs)
                return HolonAction((action[0], action[1], qknobs))
        return action

    def latent_dim(self) -> int:
        return self.raw_dims[-1]

    def critic_dim(self) -> int:
        r = [
            space.critic_dim() for space in self.spaces if hasattr(space, "critic_dim")
        ]
        assert len(r) == 3
        return sum(r)

    def get_knob_space(self) -> LatentKnobSpace:
        assert isinstance(self.spaces[0], LatentKnobSpace)
        return self.spaces[0]

    def get_index_space(self) -> LatentIndexSpace:
        assert isinstance(self.spaces[1], LatentIndexSpace)
        return self.spaces[1]

    def get_query_space(self) -> LatentQuerySpace:
        assert isinstance(self.spaces[2], LatentQuerySpace)
        return self.spaces[2]

    def pad_center_latent(self, proto: ProtoAction, lscs: torch.Tensor) -> ProtoAction:
        assert len(proto.shape) == 2

        components = []
        for i, s in enumerate(self.spaces):
            start = self.raw_dims[i - 1] if i > 0 else 0
            end = self.raw_dims[i]
            assert isinstance(s, (LatentKnobSpace, LSCIndexSpace, LatentQuerySpace))
            components.append(
                s.pad_center_latent(ProtoAction(proto[:, start:end]), lscs)
            )

        return ProtoAction(torch.cat(cast(list[torch.Tensor], components), dim=1))

    def transform_noise(
        self, proto: ProtoAction, noise: Optional[torch.Tensor] = None
    ) -> ProtoAction:
        assert len(proto.shape) == 2
        for i, s in enumerate(self.spaces):
            start = self.raw_dims[i - 1] if i > 0 else 0
            end = self.raw_dims[i]

            snoise = None
            if noise is not None:
                assert len(noise.shape) == 2
                snoise = noise[:, start:end]

            assert hasattr(s, "transform_noise")
            proto[:, start:end] = s.transform_noise(proto[:, start:end], noise=snoise)
        return proto

    def from_latent(self, proto: ProtoAction) -> ProtoAction:
        components = []
        space_dims = []
        assert len(proto.shape) == 2
        for i, s in enumerate(self.spaces):
            start = self.raw_dims[i - 1] if i > 0 else 0
            end = self.raw_dims[i]
            assert hasattr(s, "from_latent")
            components.append(s.from_latent(proto[:, start:end]))
            space_dims.append(components[-1].shape[-1])

        if self.space_dims is None:
            self.space_dims = np.cumsum(space_dims)

        return ProtoAction(torch.cat(components, dim=1))

    def to_latent(self, env_act: list[HolonAction]) -> ProtoAction:
        assert isinstance(env_act, list)
        latent_cmps = []
        for i, s in enumerate(self.spaces):
            if isinstance(s, LatentIndexSpace):
                latent_cmps.append(
                    s.to_latent(
                        cast(list[IndexSpaceRawSample], [a[i] for a in env_act])
                    )
                )
            else:
                assert isinstance(s, (LatentKnobSpace, LatentQuerySpace))
                latent_cmps.append(
                    s.to_latent(cast(list[KnobSpaceAction], [a[i] for a in env_act]))
                )

        return ProtoAction(torch.concat(cast(list[torch.Tensor], latent_cmps), dim=1))

    def sample_latent(self, mask: Optional[Any] = None) -> ProtoAction:
        r = [
            s.sample_latent(mask=mask)
            for s in self.spaces
            if hasattr(s, "sample_latent")
        ]
        assert len(r) == 3
        return ProtoAction(torch.concat(r, dim=1))

    @time_record("neighborhood")
    def neighborhood(
        self,
        raw_action: ProtoAction,
        neighbor_parameters: NeighborParameters = DEFAULT_NEIGHBOR_PARAMETERS,
    ) -> tuple[list[HolonAction], ProtoAction, torch.Tensor]:
        env_acts = []
        emb_acts: list[torch.Tensor] = []
        ndims = []

        env_action = self.from_latent(raw_action)
        for proto in env_action:
            # Figure out the neighbors for each subspace.
            envs_neighbors: list[Any] = []
            embed_neighbors: list[Any] = []

            # TODO(wz2,PROTOX_DELTA): For pseudo-backwards compatibility, we meld the knob + query space together.
            # In this way, we don't actually generate knob x query cartesian product.
            # Rather, we directly fuse min(knob_neighbors, query_neighbors) together and then cross with indexes.
            meld_groups: list[list[Any]] = [
                [self.get_knob_space(), self.get_query_space()],
                [self.get_index_space()],
            ]

            for meld_group in meld_groups:
                meld_group_envs = []
                meld_group_embeds = []
                for s in meld_group:
                    assert self.spaces.index(s) is not None
                    assert self.space_dims is not None
                    i = self.spaces.index(s)

                    # Note that subproto is the "action embedding" from the context of
                    # the embedded actor-critic like architecutre.
                    subproto = proto[
                        (self.space_dims[i - 1] if i > 0 else 0) : self.space_dims[i]
                    ]
                    assert isinstance(
                        s, (LatentKnobSpace, LatentIndexSpace, LatentQuerySpace)
                    )
                    envs = s.neighborhood(subproto, neighbor_parameters)
                    meld_group_envs.append(envs)

                    if isinstance(s, LatentIndexSpace):
                        # Compute their latent representation first.
                        meld_group_embeds.append(
                            s.to_latent(cast(list[IndexSpaceRawSample], envs))
                        )
                    else:
                        assert isinstance(s, (LatentKnobSpace, LatentQuerySpace))
                        meld_group_embeds.append(
                            s.to_latent(cast(list[KnobSpaceAction], envs))
                        )

                if len(meld_group_envs) > 1:
                    # Join the meld groups.
                    envs_neighbors.append([z for z in zip(*meld_group_envs)])
                    t_len = len(envs_neighbors[-1])
                    embed_neighbors.append(zip(*[z[:t_len] for z in meld_group_embeds]))
                else:
                    envs_neighbors.append(meld_group_envs[0])
                    embed_neighbors.append(meld_group_embeds[0])

            # Cartesian product itself is naturally in the joint space.
            carprod_neighbors = cast(
                list[HolonAction],
                [
                    (mg0[0], mg1, mg0[1])
                    for (mg0, mg1) in itertools.product(*envs_neighbors)
                ],
            )
            # Trust that the cartesian product is generated the same way.
            carprod_embeds = torch.stack(
                list(
                    map(
                        lambda mg: torch.cat((mg[0][0], mg[1], mg[0][1])),
                        itertools.product(*embed_neighbors),
                    )
                )
            )
            assert len(carprod_neighbors) == carprod_embeds.shape[0]

            # This is a sanity check to avoid having to to_latent() on each holon.
            # Guess for when the drift happens.
            first_drift = len(envs_neighbors[1]) + 1

            # Only run this check if we are attempting to sample an action and not during learn.
            if len(env_action) == 1:
                assert len(self.spaces) == len(carprod_neighbors[0])
                self._latent_assert_check(
                    carprod_neighbors, carprod_embeds, first_drift
                )

            env_acts.extend(carprod_neighbors)
            emb_acts.append(carprod_embeds)
            ndims.append(len(carprod_neighbors))

        return env_acts, ProtoAction(torch.cat(emb_acts, dim=0)), torch.as_tensor(ndims)

    def generate_state_container(
        self,
        prev_state_container: Optional[HolonStateContainer],
        action: Optional[HolonAction],
        connection: Connection[Any],
        queries: dict[str, list[tuple[QueryType, str]]],
    ) -> HolonStateContainer:
        t = tuple(
            s.generate_state_container(
                prev_state_container[i] if prev_state_container else None,
                action[i] if action else None,
                connection,
                queries,
            )
            for i, s in enumerate(self.spaces)
            if hasattr(s, "generate_state_container")
        )
        assert len(t) == 3
        return HolonStateContainer(t)

    def generate_action_plan(
        self, action: HolonAction, state_container: HolonStateContainer, **kwargs: Any
    ) -> tuple[list[str], list[str]]:
        outputs = [
            space.generate_action_plan(action[i], state_container[i], **kwargs)
            for i, space in enumerate(self.spaces)
            if hasattr(space, "generate_action_plan")
        ]
        assert len(outputs) == 3
        cc = list(itertools.chain(*[o[0] for o in outputs]))
        sql_commands = list(itertools.chain(*[o[1] for o in outputs]))
        return cc, sql_commands

    def generate_plan_from_config(
        self, config: HolonStateContainer, sc: HolonStateContainer, **kwargs: Any
    ) -> tuple[list[str], list[str]]:
        outputs = [
            space.generate_delta_action_plan(config[i], sc[i], **kwargs)
            for i, space in enumerate(self.spaces)
            if hasattr(space, "generate_delta_action_plan")
        ]
        assert len(outputs) == 3
        config_changes = list(itertools.chain(*[o[0] for o in outputs]))
        sql_commands = list(itertools.chain(*[o[1] for o in outputs]))
        return config_changes, sql_commands
