from typing import Any, List, Optional

import gymnasium as gym
import torch
from gymnasium import spaces

from tune.protox.env.space.primitive.index import IndexAction
from tune.protox.env.space.primitive_space.index_policy import IndexPolicy
from tune.protox.env.types import (
    IndexSpaceRawSample,
    TableAttrAccessSetsMap,
    TableAttrListMap,
    TableColTuple,
)


class IndexSpace(spaces.Tuple):
    def get_index_class(self, env_act: IndexSpaceRawSample) -> str:
        ia = self.to_action(env_act)
        if not ia.is_valid:
            return "-1"
        return str(self.class_mapping[TableColTuple((ia.tbl_name, ia.columns[0]))])

    def __init__(
        self,
        tables: list[str],
        max_num_columns: int,
        max_indexable_attributes: int,
        seed: int,
        rel_metadata: TableAttrListMap,
        attributes_overwrite: TableAttrListMap,
        tbl_include_subsets: TableAttrAccessSetsMap,
        index_space_aux_type: bool = False,
        index_space_aux_include: bool = False,
        deterministic_policy: bool = False,
    ):

        self.max_num_columns = self.max_inc_columns = max_num_columns
        self.rel_metadata = rel_metadata
        self.tbl_include_subsets = tbl_include_subsets
        self.index_space_aux_type = index_space_aux_type
        self.index_space_aux_include = index_space_aux_include

        if attributes_overwrite is not None:
            # Overwrite the maximum number of columns.
            self.max_num_columns = max_indexable_attributes
            for k, v in attributes_overwrite.items():
                # Overwrite and substitute.
                self.rel_metadata[f"{k}_allcols"] = self.rel_metadata[k]
                self.rel_metadata[k] = v

        self.tables = tables
        self.policy = IndexPolicy(
            tables,
            self.rel_metadata,
            self.tbl_include_subsets,
            self.max_num_columns,
            max_num_columns,
            index_space_aux_type,
            index_space_aux_include,
            deterministic_policy,
        )

        # Create class mapping.
        self.class_mapping: dict[TableColTuple, int] = {}
        for tbl in self.tables:
            for col in rel_metadata[tbl]:
                self.class_mapping[TableColTuple((tbl, col))] = len(self.class_mapping)

        super().__init__(spaces=self.policy.spaces(seed), seed=seed)

    def to_action(self, act: IndexSpaceRawSample) -> IndexAction:
        return self.policy.to_action(act)

    def sample(self, mask: Optional[Any] = None) -> IndexSpaceRawSample:
        table_idx = None if mask is None else mask.get("table_idx", None)
        col_idx = None if mask is None else mask.get("col_idx", None)
        action = torch.zeros(gym.spaces.utils.flatdim(self))
        assert not self.index_space_aux_type and not self.index_space_aux_include
        if table_idx is None:
            # Make equal weight.
            action[0 : len(self.tables)] = 1.0 / len(self.tables)
        else:
            # Hit only the targeted table.
            action[table_idx] = 1

        if col_idx is not None:
            action[len(self.tables) + col_idx + 1] = 1.0
            # Evenly distribute the column weights.
            action[len(self.tables) + (self.max_num_columns + 1) :] = 1.0 / (
                self.max_num_columns + 1
            )
        else:
            action[len(self.tables) :] = 1.0 / (self.max_num_columns + 1)

        return self.policy.sample_dist(
            action, self.np_random, sample_num_columns=True, break_early=False
        )

    def null_action(self) -> IndexSpaceRawSample:
        action = torch.zeros(gym.spaces.utils.flatdim(self))
        if self.index_space_aux_type:
            action = action[2:]
        if self.index_space_aux_include:
            action = action[: -self.max_inc_columns]

        action[0] = 1.0
        return self.policy.sample_dist(action, self.np_random, sample_num_columns=False)

    def to_jsonable(self, sample_n) -> List[str]:  # type: ignore
        # Emit the representation of an index.
        ias = [self.to_action(sample) for sample in sample_n]
        return [ia.__repr__() for ia in ias]
