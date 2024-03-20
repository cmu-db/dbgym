import copy
from typing import Any, List, Sequence, cast

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.spaces import Box
from torch.nn.functional import softmax

from tune.protox.env.space.primitive.index import IndexAction
from tune.protox.env.types import (
    IndexSpaceRawSample,
    TableAttrAccessSetsMap,
    TableAttrListMap,
)


class IndexPolicy:
    def __init__(
        self,
        tables: list[str],
        rel_metadata: TableAttrListMap,
        tbl_include_subsets: TableAttrAccessSetsMap,
        max_key_columns: int,
        max_num_columns: int,
        index_space_aux_type: bool = False,
        index_space_aux_include: bool = False,
        deterministic: bool = False,
    ):

        self.tables = tables
        self.rel_metadata = rel_metadata
        self.tbl_include_subsets = tbl_include_subsets
        self.num_tables = len(self.tables)
        self.max_key_columns = max_key_columns
        self.max_num_columns = max_num_columns
        self.deterministic = deterministic

        self.num_index_types = 2
        self.index_types = ["btree", "hash"]
        self.index_space_aux_type = index_space_aux_type
        self.index_space_aux_include = index_space_aux_include

    def spaces(self, seed: int) -> Sequence[spaces.Space[Any]]:
        aux: List[spaces.Space[Any]] = [
            # One-hot encoding for the tables.
            spaces.Discrete(self.num_tables, seed=seed),
            # Ordering. Note that we use the postgres style ordinal notation. 0 is illegal/end-of-index.
            *(
                [spaces.Discrete(self.max_key_columns + 1, seed=seed)]
                * self.max_key_columns
            ),
        ]
        aux_include = []
        aux_type = []

        if self.index_space_aux_type:
            aux_type = [spaces.Discrete(self.num_index_types, seed=seed)]

        if self.index_space_aux_include > 0:
            aux_include = [
                Box(
                    low=np.zeros(self.max_num_columns),
                    high=1.0,
                    seed=seed,
                    dtype=np.float32,
                )
            ]

        return cast(List[spaces.Space[Any]], aux_type + aux + aux_include)

    def to_action(self, act: IndexSpaceRawSample) -> IndexAction:
        # First index is the index type.
        tbl_name = (
            self.tables[act[1]] if self.index_space_aux_type else self.tables[act[0]]
        )
        idx_type = 0
        inc_cols: list[int] = []
        if self.index_space_aux_type and self.index_space_aux_include:
            idx_type = act[0]
            columns = act[2:-1]
            inc_cols = act[-1]
        elif self.index_space_aux_type:
            idx_type = act[0]
            columns = act[2:]
        elif self.index_space_aux_include:
            columns = act[1:-1]
            inc_cols = act[-1]
        else:
            columns = act[1:]

        col_names = []
        col_idxs = []
        for i in columns:
            if i == 0:
                break
            # Create the index key.
            col_names.append(self.rel_metadata[tbl_name][i - 1])
            col_idxs.append(i - 1)

        if len(inc_cols) > 0:
            # Get all include columns.
            assert f"{tbl_name}_allcols" in self.rel_metadata
            valid_names = [n for n in self.rel_metadata[f"{tbl_name}_allcols"]]
            inc_names = [
                valid_names[i]
                for i, val in enumerate(inc_cols)
                if val == 1.0 and valid_names[i] not in col_names
            ]
        else:
            inc_names = []

        return IndexAction(
            self.index_types[idx_type],
            tbl_name,
            col_names,
            col_idxs,
            inc_names,
            act,
            bias=0,
        )

    def sample_dist(
        self,
        action: torch.Tensor,
        np_random: np.random.Generator,
        sample_num_columns: bool = False,
        break_early: bool = True,
    ) -> IndexSpaceRawSample:
        # Acquire the table index either deterministically or not.
        if self.deterministic:
            tbl_index = torch.argmax(action[: self.num_tables]).item()
        else:
            tbl_index = torch.multinomial(action[: self.num_tables], 1).item()

        # Get the number of columns.
        num_columns = len(self.rel_metadata[self.tables[int(tbl_index)]])
        use_columns = num_columns
        if sample_num_columns:
            # If we sample columns, sample it.
            use_columns = np_random.integers(1, num_columns + 1)

        # Prune off.
        action = action[self.num_tables :]

        assert len(action.shape) == 1
        action = action.clone()
        action = action.reshape((self.max_key_columns, self.max_key_columns + 1))
        action = action[:, 0 : num_columns + 1]

        if not break_early:
            # Zero out the early break odds.
            action[:, 0] = 0

        current_index = 0
        col_indexes: list[int] = []
        while current_index < action.shape[0] and len(col_indexes) != use_columns:
            if not torch.any(action[current_index]):
                # No more positive probability to sample.
                break

            # Acquire a column index depending on determinism or not.
            if self.deterministic:
                col_index = int(torch.argmax(action[current_index]).item())
            else:
                col_index = int(torch.multinomial(action[current_index], 1).item())

            if break_early and col_index == 0:
                # We've explicitly decided to terminate it early.
                break

            # Directly use the col_index. Observe that "0" is the illegal.
            if col_index not in col_indexes:
                action[:, col_index] = 0
                col_indexes.append(col_index)

            # Always advance since we don't let you take duplicates.
            current_index += 1

        np_col_indexes = np.pad(
            np.array(col_indexes),
            [0, self.max_key_columns - len(col_indexes)],
            mode="constant",
            constant_values=0,
        ).astype(int)
        if self.index_space_aux_type and self.index_space_aux_include:
            return IndexSpaceRawSample(
                (
                    0,
                    tbl_index,
                    *np_col_indexes,
                    np.array([0] * self.max_num_columns, dtype=np.float32),
                )
            )
        elif self.index_space_aux_include:
            return IndexSpaceRawSample(
                (
                    tbl_index,
                    *np_col_indexes,
                    np.array([0] * self.max_num_columns, dtype=np.float32),
                )
            )
        elif self.index_space_aux_type:
            return IndexSpaceRawSample((0, tbl_index, *np_col_indexes))
        else:
            return IndexSpaceRawSample((tbl_index, *np_col_indexes))

    def structural_neighbors(
        self, action: IndexSpaceRawSample
    ) -> list[IndexSpaceRawSample]:
        idx_type = 0
        inc_columns: list[int] = []
        if self.index_space_aux_type and self.index_space_aux_include:
            tbl_index = action[1]
            columns = action[2:-1]
            inc_columns = action[-1]
        elif self.index_space_aux_type:
            tbl_index = action[1]
            columns = action[2:]
        elif self.index_space_aux_include:
            tbl_index = action[0]
            columns = action[1:-1]
            inc_columns = action[-1]
        else:
            tbl_index = action[0]
            columns = action[1:]

        num_columns = len(columns)
        new_candidates = [action]

        # Generate the "prefix rule".
        for i in range(len(columns)):
            # No more valid indexes to construct.
            if columns[i] == 0:
                break

            # Construct prefix index of the current index.
            new_columns = [0 for _ in range(num_columns)]
            new_columns[: i + 1] = columns[: i + 1]

            if self.index_space_aux_type and self.index_space_aux_include:
                act = (idx_type, tbl_index, *new_columns, inc_columns)
            elif self.index_space_aux_type:
                act = (idx_type, tbl_index, *new_columns)
            elif self.index_space_aux_include:
                act = (tbl_index, *new_columns, inc_columns)
            else:
                act = (tbl_index, *new_columns)
            new_candidates.append(IndexSpaceRawSample(act))

        # Generate "index type" rule.
        if self.index_space_aux_type:
            hash_act = list(copy.deepcopy(action))
            hash_act[0] = 1
            for i in range(3, 2 + num_columns):
                hash_act[i] = 0
            new_candidates.append(IndexSpaceRawSample(tuple(hash_act)))

        # Generate "include" rule.
        if self.index_space_aux_include and self.tbl_include_subsets:
            inc_subsets = self.tbl_include_subsets[self.tables[tbl_index]]
            aux_candidates = []
            for candidate in new_candidates:
                if self.index_space_aux_type:
                    if candidate[0] == 1:
                        # This is a HASH()
                        continue
                    columns = candidate[2:-1]
                else:
                    columns = candidate[1:-1]

                names = [
                    self.rel_metadata[self.tables[tbl_index]][col - 1]
                    for col in columns
                    if col > 0
                ]
                for inc_subset in inc_subsets:
                    inc_cols = [s for s in inc_subset if s not in names]
                    if len(inc_cols) > 0:
                        # Construct the bit flag map.
                        flag = np.zeros(self.max_num_columns, dtype=np.float32)
                        for inc_col in inc_cols:
                            flag[
                                self.rel_metadata[
                                    f"{self.tables[tbl_index]}_allcols"
                                ].index(inc_col)
                            ] = 1
                        aux_candidates.append(
                            IndexSpaceRawSample((*candidate[:-1], flag))
                        )
            new_candidates.extend(aux_candidates)
        return new_candidates

    def from_latent(self, subproto: torch.Tensor) -> torch.Tensor:
        num_tables = self.num_tables
        max_cols = self.max_key_columns
        assert len(subproto.shape) == 2

        # First apply the softmax.
        subproto[:, :num_tables] = softmax(subproto[:, :num_tables], dim=1)
        # Now apply the per ordinal softmax.
        x_reshape = subproto[:, num_tables:].reshape(
            subproto.shape[0], max_cols, max_cols + 1
        )
        x_reshape = softmax(x_reshape, dim=2)
        subproto[:, num_tables:] = x_reshape.reshape(subproto.shape[0], -1)
        return subproto
