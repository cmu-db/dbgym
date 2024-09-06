from typing import Any, Callable, Optional, Tuple

import numpy as np
import psycopg
import torch
from numpy.typing import NDArray

from tune.protox.embedding.vae import VAE
from tune.protox.env.artifact_manager import ArtifactManager, time_record
from tune.protox.env.space.primitive.index import IndexAction
from tune.protox.env.space.primitive_space import IndexSpace
from tune.protox.env.space.utils import check_subspace, fetch_server_indexes
from tune.protox.env.types import (
    DEFAULT_NEIGHBOR_PARAMETERS,
    IndexSpaceContainer,
    IndexSpaceRawSample,
    NeighborParameters,
    ProtoAction,
    QueryMap,
    TableAttrAccessSetsMap,
    TableAttrListMap,
)


class LatentIndexSpace(IndexSpace):
    def __init__(
        self,
        tables: list[str],
        max_num_columns: int,
        max_indexable_attributes: int,
        seed: int,
        rel_metadata: TableAttrListMap,
        attributes_overwrite: TableAttrListMap,
        tbl_include_subsets: TableAttrAccessSetsMap,
        vae: VAE,
        index_space_aux_type: bool = False,
        index_space_aux_include: bool = False,
        deterministic_policy: bool = False,
        latent_dim: int = 0,
        index_output_transform: Optional[Callable[[ProtoAction], ProtoAction]] = None,
        index_noise_scale: Optional[
            Callable[[ProtoAction, Optional[torch.Tensor]], ProtoAction]
        ] = None,
        artifact_manager: Optional[ArtifactManager] = None,
    ) -> None:

        super().__init__(
            tables,
            max_num_columns,
            max_indexable_attributes,
            seed,
            rel_metadata,
            attributes_overwrite,
            tbl_include_subsets,
            index_space_aux_type,
            index_space_aux_include,
            deterministic_policy,
        )

        self.vae = vae
        self._latent_dim = latent_dim
        self.index_output_transform = index_output_transform
        self.index_noise_scale = index_noise_scale
        self.artifact_manager = artifact_manager
        self.name = "index"

    def latent_dim(self) -> int:
        return self._latent_dim

    def critic_dim(self) -> int:
        index_dim = self.latent_dim()
        if self.index_space_aux_type:
            index_dim += 2

        if self.index_space_aux_include:
            index_dim += self.max_inc_columns

        return index_dim

    def uses_embed(self) -> bool:
        return True

    def transform_noise(
        self, subproto: ProtoAction, noise: Optional[torch.Tensor] = None
    ) -> ProtoAction:
        if self.index_output_transform is not None:
            subproto = self.index_output_transform(subproto)
        else:
            subproto = ProtoAction(torch.tanh(subproto))

        if noise is not None and self.index_noise_scale:
            # Now perturb noise.
            subproto = self.index_noise_scale(subproto, noise)

        return subproto

    @time_record("from_latent")
    def from_latent(self, subproto: ProtoAction) -> ProtoAction:
        assert len(subproto.shape) == 2
        decode_act = self.vae.decoder(subproto).detach().view(subproto.shape[0], -1)

        # Only need to do additional processing if we are treating as one-hot softmax representation.
        # Now treat it as it came out of the neural network and process it.
        if len(self.tables) < self.max_num_columns + 1:
            # Yoink only the table components that we care about.
            distort = [l for l in range(0, len(self.tables))] + [
                l for l in range(self.max_num_columns + 1, decode_act.shape[1])
            ]
        else:
            # Yoink only the index components that we care about.
            distort = [l for l in range(0, len(self.tables))]
            [
                distort.extend([b + i for i in range(0, self.max_num_columns + 1)])  # type: ignore
                for b in range(len(self.tables), decode_act.shape[1], len(self.tables))
            ]

        decode_act = torch.index_select(decode_act, 1, torch.as_tensor(distort))
        return ProtoAction(self.policy.from_latent(decode_act))

    @time_record("to_latent")
    def to_latent(self, env_act: list[IndexSpaceRawSample]) -> ProtoAction:
        assert isinstance(env_act, list)
        if self.index_space_aux_include:
            # Straighten out the list.
            th_env_act = torch.as_tensor(
                [(e[:-1]) + tuple(e[-1]) for e in env_act]
            ).reshape(len(env_act), -1)
        else:
            th_env_act = torch.as_tensor(env_act).reshape(len(env_act), -1)

        index_type = None
        include_col = None
        if self.index_space_aux_type:
            # Boink the index type.
            index_val = th_env_act[:, 0].view(th_env_act.shape[0], -1).type(torch.int64)
            index_type = torch.zeros(index_val.shape[0], 2, dtype=torch.int64)
            index_type = index_type.scatter_(1, index_val, 1).float()
            th_env_act = th_env_act[:, 1:]

        if self.index_space_aux_include:
            include_col = th_env_act[:, -self.max_inc_columns :].float()
            th_env_act = th_env_act[:, : -self.max_inc_columns]

        nets = self.vae.get_collate()(th_env_act).float()

        # There isn't much we can do if we encounter an error.
        latents, error = self.vae.latents(nets)
        assert not error

        if index_type is not None:
            latents = torch.concat([index_type, latents], dim=1)
        if include_col is not None:
            latents = torch.concat([latents, include_col], dim=1)

        return ProtoAction(latents.float())

    def sample_latent(self, mask: Optional[Any] = None) -> NDArray[np.float32]:
        return (
            np.random.uniform(low=0.0, high=1.0, size=(self.latent_dim(),))
            .astype(np.float32)
            .reshape(1, -1)
        )

    @time_record("neighborhood")
    def neighborhood(
        self,
        raw_action: ProtoAction,
        neighbor_parameters: NeighborParameters = DEFAULT_NEIGHBOR_PARAMETERS,
    ) -> list[IndexSpaceRawSample]:
        actions_set = set()
        actions: list[IndexSpaceRawSample] = []

        num_attempts = 0
        allow_random_samples = False
        while len(actions) == 0:
            for _ in range(neighbor_parameters["index_num_samples"]):
                sampled_action = self.policy.sample_dist(raw_action, self.np_random)
                assert self.contains(sampled_action)
                candidates = [sampled_action]

                if allow_random_samples:
                    # Only allow *pure* random samples once the flag has been set.
                    random_act = self.sample()
                    assert self.contains(random_act)
                    candidates.append(random_act)

                # Sample subsets if we aren't sampling the length of the index already.
                if neighbor_parameters["index_rules"]:
                    for candidate in self.policy.structural_neighbors(sampled_action):
                        assert self.contains(candidate)
                        candidates.append(candidate)

                for candidate in candidates:
                    ia = self.to_action(candidate)
                    # Make sure that we are setting the bias + raw representation.
                    assert ia.bias is not None and ia.raw_repr is not None

                    if ia not in actions_set:
                        # See IndexAction.__hash__ comment.
                        actions.append(candidate)
                        actions_set.add(ia)

                if len(actions) >= neighbor_parameters["index_num_samples"]:
                    # We have generated enough candidates. At least based on num_samples.
                    break

            num_attempts += 1
            if num_attempts >= 100:
                # Log but don't crash.
                if self.artifact_manager:
                    self.artifact_manager.get_logger(__name__).error(
                        "Spent 100 iterations and could not find any valid index action. This should not happen."
                    )
                allow_random_samples = True
        return actions

    def generate_state_container(
        self,
        prev_state_container: Optional[IndexSpaceContainer],
        action: Optional[IndexSpaceRawSample],
        connection: psycopg.Connection[Any],
        queries: QueryMap,
    ) -> IndexSpaceContainer:

        ias = []
        _, indexes = fetch_server_indexes(connection, self.tables)
        for tblname, indexdefs in indexes.items():
            for idxname, indexdef in indexdefs.items():
                ias.append(
                    IndexAction.construct_md(
                        idx_name=idxname,
                        table=tblname,
                        idx_type=indexdef["index_type"],
                        columns=indexdef["columns"],
                        inc_names=indexdef["include"],
                    )
                )

        new_ia: Optional[IndexAction] = None
        if action:
            new_ia = self.to_action(action)

        for ia in ias:
            if prev_state_container and ia in prev_state_container:
                p = prev_state_container[prev_state_container.index(ia)]
                ia.raw_repr = p.raw_repr
            elif new_ia and ia == new_ia:
                ia.raw_repr = new_ia.raw_repr
        return IndexSpaceContainer(ias)

    def generate_action_plan(
        self, action: IndexSpaceRawSample, sc: IndexSpaceContainer, **kwargs: Any
    ) -> tuple[list[str], list[str]]:
        assert check_subspace(self, action)

        sql_commands = []
        ia = self.to_action(action)
        if not ia.is_valid:
            # This is the case where the action we are taking is a no-op.
            return [], []

        exist_ia = ia in sc
        if exist_ia:
            if self.artifact_manager:
                self.artifact_manager.get_logger(__name__).debug(
                    "Contemplating %s (exist: True)", sc[sc.index(ia)]
                )
        else:
            if self.artifact_manager:
                self.artifact_manager.get_logger(__name__).debug(
                    "Contemplating %s (exist: False)", ia
                )
            # Add the new index with the current index counter.
            sql_commands.append(ia.sql(add=True))

        return [], sql_commands

    def generate_delta_action_plan(
        self, action: IndexSpaceContainer, sc: IndexSpaceContainer, **kwargs: Any
    ) -> tuple[list[str], list[str]]:
        assert isinstance(action, list)
        acts = []
        sql_commands = []
        for ia in action:
            assert isinstance(ia, IndexAction)
            acts.append(ia)

            if not ia.is_valid:
                # This is the case where the action we are taking is a no-op.
                continue

            if ia not in sc:
                # Create if not exist.
                sql_commands.append(ia.sql(add=True))

        for ia in sc:
            # Drop the index that is no longer needed.
            if ia not in acts:
                sql_commands.append(ia.sql(add=False))

        return [], sql_commands

    def to_action(self, env_act: IndexSpaceRawSample) -> IndexAction:
        ia = super().to_action(env_act)
        ia.raw_repr = env_act
        return ia
