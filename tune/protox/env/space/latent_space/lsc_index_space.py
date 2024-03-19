from typing import Any, Callable, Optional
import psycopg
import torch

from tune.protox.embedding.vae import VAE
from tune.protox.env.logger import Logger
from tune.protox.env.lsc import LSC
from tune.protox.env.space.latent_space.latent_index_space import LatentIndexSpace
from tune.protox.env.space.primitive.index import IndexAction
from tune.protox.env.types import (
    IndexSpaceContainer,
    IndexSpaceRawSample,
    ProtoAction,
    QueryMap,
    TableAttrAccessSetsMap,
    TableAttrListMap,
)


class LSCIndexSpace(LatentIndexSpace):
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
            Callable[[ProtoAction, torch.Tensor], ProtoAction]
        ] = None,
        logger: Optional[Logger] = None,
        lsc: Optional[LSC] = None,
    ) -> None:

        super().__init__(
            tables,
            max_num_columns,
            max_indexable_attributes,
            seed,
            rel_metadata,
            attributes_overwrite,
            tbl_include_subsets,
            vae,
            index_space_aux_type,
            index_space_aux_include,
            deterministic_policy,
            latent_dim,
            index_output_transform,
            index_noise_scale,
            logger,
        )

        assert lsc is not None
        self.lsc = lsc

    def pad_center_latent(
        self, subproto: ProtoAction, lscs: torch.Tensor
    ) -> ProtoAction:
        subproto = ProtoAction(subproto + self.lsc.inverse_scale(lscs))

        if self.index_space_aux_type:
            aux_types = torch.tensor([[1.0, 0.0]] * subproto.shape[0]).float()
            subproto = ProtoAction(torch.concat([aux_types, subproto], dim=1))

        if self.index_space_aux_include:
            aux_inc = torch.tensor(
                [[0] * self.max_inc_columns] * subproto.shape[0]
            ).float()
            subproto = ProtoAction(torch.concat([subproto, aux_inc], dim=1))
        return subproto

    def from_latent(self, subproto: ProtoAction) -> ProtoAction:
        subproto = self.lsc.apply_bias(subproto)
        return super().from_latent(subproto)

    def to_latent(self, env_act: list[IndexSpaceRawSample]) -> ProtoAction:
        latent = super().to_latent(env_act)
        assert len(latent.shape) == 2

        if latent.shape[-1] != self.latent_dim():
            so = 2 if self.index_space_aux_type else 0
            latent[:, so:so+self.latent_dim()] = self.lsc.apply_bias(ProtoAction(latent[:, so:so + self.latent_dim()]))
            return latent
        else:
            return self.lsc.apply_bias(latent)

    def generate_state_container(
        self,
        prev_state: Optional[IndexSpaceContainer],
        action: Optional[IndexSpaceRawSample],
        connection: psycopg.Connection[Any],
        queries: QueryMap,
    ) -> IndexSpaceContainer:
        ias = super().generate_state_container(prev_state, action, connection, queries)

        new_ia: Optional[IndexAction] = None
        if action:
            new_ia = self.to_action(action)

        for ia in ias:
            if prev_state and ia in prev_state:
                # Preserve the bias.
                ia.bias = prev_state[prev_state.index(ia)].bias

            elif new_ia and ia == new_ia:
                ia.bias = new_ia.bias

        return ias

    def to_action(self, env_act: IndexSpaceRawSample) -> IndexAction:
        ia = super().to_action(env_act)
        ia.bias = self.lsc.current_bias()
        return ia
