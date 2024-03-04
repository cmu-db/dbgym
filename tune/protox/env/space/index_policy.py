import copy
from abc import ABC, abstractmethod
from enum import Enum, unique

import numpy as np
import torch
from gymnasium import spaces
from torch.nn.functional import softmax

from tune.protox.env.space.real import Real


def _sample_onehot_subsets(
    action,
    index_space_aux_type,
    index_space_aux_include,
    rel_metadata=None,
    tables=None,
    tbl_include_subsets=None,
    column_ordinal_mask=None,
):

    idx_type = 0
    inc_columns = []
    if index_space_aux_type and index_space_aux_include > 0:
        tbl_index = action[1]
        columns = action[2:-index_space_aux_include]
        inc_columns = action[-index_space_aux_include:]
    elif index_space_aux_type:
        tbl_index = action[1]
        columns = action[2:]
    elif index_space_aux_include > 0:
        tbl_index = action[0]
        columns = action[1:-index_space_aux_include]
        inc_columns = action[-index_space_aux_include:]
    else:
        tbl_index = action[0]
        columns = action[1:]

    num_columns = len(columns)
    if column_ordinal_mask is not None:
        # If the ordinal value isn't accepted by the mask, then skip it.
        columns = [c for c in columns if (c - 1) in column_ordinal_mask or c == 0]

    new_candidates = [action]
    for i in range(len(columns)):
        # No more valid indexes to construct.
        if columns[i] == 0:
            break

        # Construct prefix index of the current index.
        new_columns = [0 for _ in range(num_columns)]
        new_columns[: i + 1] = columns[: i + 1]

        if index_space_aux_type and index_space_aux_include > 0:
            act = (idx_type, tbl_index, *new_columns, *inc_columns)
        elif index_space_aux_type:
            act = (idx_type, tbl_index, *new_columns)
        elif index_space_aux_include > 0:
            act = (tbl_index, *new_columns, *inc_columns)
        else:
            act = (tbl_index, *new_columns)
        new_candidates.append(act)

    if index_space_aux_type:
        hash_act = list(copy.deepcopy(action))
        hash_act[0] = 1
        for i in range(3, 2 + num_columns):
            hash_act[i] = 0
        new_candidates.append(tuple(hash_act))

    if index_space_aux_include > 0 and tbl_include_subsets is not None:
        assert "actual" in rel_metadata
        inc_subsets = tbl_include_subsets[tables[tbl_index]]
        aux_candidates = []
        for candidate in new_candidates:
            if index_space_aux_type:
                if candidate[0] == 1:
                    # This is a HASH()
                    continue
                columns = candidate[2:-index_space_aux_include]
            else:
                columns = candidate[1:-index_space_aux_include]

            names = [
                rel_metadata[tables[tbl_index]][col - 1] for col in columns if col > 0
            ]
            for inc_subset in inc_subsets:
                inc_cols = [s for s in inc_subset if s not in names]
                if len(inc_cols) > 0:
                    # Construct the bit flag map.
                    flag = [0] * index_space_aux_include
                    for inc_col in inc_cols:
                        flag[
                            rel_metadata["actual"][tables[tbl_index]].index(inc_col)
                        ] = 1
                    aux_candidates.append(
                        (*candidate[:-index_space_aux_include], *flag)
                    )
        new_candidates.extend(aux_candidates)
    return new_candidates


@unique
class IndexRepr(Enum):
    # <one-hot table, dist over cols for col1, dist over cols for col2, ...> w. sampling
    ONE_HOT = 0
    # <one-hot table, dist over cols for col1, dist over cols for col2, ...> w. argmax at each step
    # This is equivalently an auto-reg like policy.
    ONE_HOT_DETERMINISTIC = 1


class IndexPolicy(ABC):
    def __init__(self, tables, max_num_columns, deterministic):
        self.tables = tables
        self.num_tables = len(self.tables)
        self.max_num_columns = max_num_columns
        self.deterministic = deterministic

        self.num_index_types = 2
        self.index_types = ["btree", "hash"]

    @abstractmethod
    def process_network_output(self, subproto):
        pass

    @abstractmethod
    def perturb_noise(self, proto, noise):
        pass

    @abstractmethod
    def act_to_columns(self, act, rel_metadata):
        pass

    def sample_action(
        self,
        np_random,
        action,
        rel_metadata,
        sample_num_columns,
        allow_break=True,
        column_override=None,
    ):
        # Acquire the table index either deterministically or not.
        if self.deterministic:
            tbl_index = torch.argmax(action[: self.num_tables]).item()
        else:
            tbl_index = torch.multinomial(action[: self.num_tables], 1).item()

        # Get the number of columns.
        num_columns = len(rel_metadata[self.tables[tbl_index]])
        use_columns = num_columns
        if column_override:
            use_columns = column_override
        elif sample_num_columns:
            # If we sample columns, sample it.
            use_columns = np_random.integers(1, num_columns + 1)

        return self._sample_action(
            np_random,
            tbl_index,
            action[self.num_tables :],
            num_columns,
            use_columns,
            allow_break=allow_break,
        )

    @abstractmethod
    def _sample_action(
        self, np_random, tbl_index, action, num_columns, use_columns, allow_break=True
    ):
        pass

    @abstractmethod
    def sample_subsets(
        self, action, tbl_include_subsets=None, column_ordinal_mask=None
    ):
        pass

    @abstractmethod
    def spaces(self, num_tables, seed):
        pass

    def index_logits(self, tbl_name, col_names, rel_metadata):
        assert False


class OneHotIndexPolicy(IndexPolicy):
    def __init__(
        self,
        tables,
        max_num_columns,
        index_space_aux_type,
        index_space_aux_include,
        maximize=False,
    ):
        super().__init__(tables, max_num_columns, deterministic=maximize)
        self.index_space_aux_type = index_space_aux_type
        self.index_space_aux_include = index_space_aux_include

    def act_to_columns(self, act, rel_metadata):
        tbl_name = (
            self.tables[act[1]] if self.index_space_aux_type else self.tables[act[0]]
        )
        idx_type = 0
        inc_cols = []
        if self.index_space_aux_type and self.index_space_aux_include > 0:
            idx_type = act[0]
            columns = act[2 : -self.index_space_aux_include]
            inc_cols = act[-self.index_space_aux_include :]
        elif self.index_space_aux_type:
            idx_type = act[0]
            columns = act[2:]
        elif self.index_space_aux_include > 0:
            columns = act[1 : -self.index_space_aux_include]
            inc_cols = act[-self.index_space_aux_include :]
        else:
            columns = act[1:]

        col_names = []
        col_idxs = []
        for i in columns:
            if i == 0:
                break
            col_names.append(rel_metadata[tbl_name][i - 1])
            col_idxs.append(i - 1)

        if len(inc_cols) > 0:
            assert "actual" in rel_metadata
            valid_names = [n for n in rel_metadata["actual"][tbl_name]]
            inc_names = [
                valid_names[i]
                for i, val in enumerate(inc_cols)
                if val == 1.0 and valid_names[i] not in col_names
            ]
        else:
            inc_names = []

        return self.index_types[idx_type], tbl_name, col_names, col_idxs, inc_names

    def _sample_action(
        self, np_random, tbl_index, action, num_columns, use_columns, allow_break=True
    ):
        assert len(action.shape) == 1
        action = action.clone()
        action = action.reshape((self.max_num_columns, self.max_num_columns + 1))
        action = action[:, 0 : num_columns + 1]

        if not allow_break:
            # Zero out the odds.
            action[:, 0] = 0

        current_index = 0
        col_indexes = []
        while current_index < action.shape[0] and len(col_indexes) != use_columns:
            if not torch.any(action[current_index]):
                # No more positive probability to sample.
                break

            # Acquire a column index depending on determinism or not.
            if self.deterministic:
                col_index = torch.argmax(action[current_index]).item()
            else:
                col_index = torch.multinomial(action[current_index], 1).item()

            if allow_break and col_index == 0:
                # We've explicitly decided to terminate it early.
                break

            # Directly use the col_index. Observe that "0" is the illegal.
            if col_index not in col_indexes:
                action[:, col_index] = 0
                col_indexes.append(col_index)

            # Always advance since we don't let you take duplicates.
            current_index += 1

        col_indexes = np.pad(
            col_indexes,
            (0, self.max_num_columns - len(col_indexes)),
            mode="constant",
            constant_values=0,
        )
        col_indexes = col_indexes.astype(int)
        if self.index_space_aux_type and self.index_space_aux_include > 0:
            return (0, tbl_index, *col_indexes, *([0] * self.index_space_aux_include))
        elif self.index_space_aux_include > 0:
            return (tbl_index, *col_indexes, *([0] * self.index_space_aux_include))
        elif self.index_space_aux_type:
            return (0, tbl_index, *col_indexes)
        else:
            return (tbl_index, *col_indexes)

    def sample_subsets(
        self,
        action,
        rel_metadata=None,
        tbl_include_subsets=None,
        column_ordinal_mask=None,
    ):
        return _sample_onehot_subsets(
            action,
            self.index_space_aux_type,
            self.index_space_aux_include,
            rel_metadata=rel_metadata,
            tables=self.tables,
            tbl_include_subsets=tbl_include_subsets,
            column_ordinal_mask=column_ordinal_mask,
        )

    def spaces(self, seed):
        aux_type = []
        aux = [
            # One-hot encoding for the tables.
            spaces.Discrete(self.num_tables, seed=seed),
            # Ordering. Note that we use the postgres style ordinal notation. 0 is illegal/end-of-index.
            *(
                [spaces.Discrete(self.max_num_columns + 1, seed=seed)]
                * self.max_num_columns
            ),
        ]
        aux_include = []

        if self.index_space_aux_type:
            aux_type = [spaces.Discrete(self.num_index_types, seed=seed)]

        if self.index_space_aux_include > 0:
            aux_include = [
                Real(low=0.0, high=1.0, seed=seed, dtype=np.float32)
            ] * self.index_space_aux_include

        return aux_type + aux + aux_include

    def process_network_output(self, subproto, select=True):
        num_tables = self.num_tables
        max_cols = self.max_num_columns
        if not select:
            row_len = max(num_tables, max_cols + 1)
            if len(subproto.shape) == 2:
                subproto[:, :row_len] = softmax(subproto[:, :row_len], dim=1)
                r = subproto[:, row_len:].reshape(subproto.shape[0], max_cols, row_len)
                r = softmax(r, dim=2)
                subproto[:, row_len:] = r.reshape(subproto.shape[0], -1)
                return subproto
            else:
                subproto[:row_len] = softmax(subproto[:row_len], dim=0)
                r = subproto[row_len:].reshape(max_cols, row_len)
                r = softmax(r, dim=1)
                subproto[row_len:] = r.reshape(subproto.shape[0], -1)
                return subproto

        if len(subproto.shape) == 2:
            # First apply the softmax.
            subproto[:, :num_tables] = softmax(subproto[:, :num_tables], dim=1)
            # Now apply the per ordinal softmax.
            x_reshape = subproto[:, num_tables:].reshape(
                subproto.shape[0], max_cols, max_cols + 1
            )
            x_reshape = softmax(x_reshape, dim=2)
            subproto[:, num_tables:] = x_reshape.reshape(subproto.shape[0], -1)
        else:
            # First apply the softmax.
            subproto[:num_tables] = softmax(subproto[:num_tables], dim=0)
            # Now apply the per ordinal softmax.
            x_reshape = subproto[num_tables:].reshape(max_cols, max_cols + 1)
            x_reshape = softmax(x_reshape, dim=1)
            subproto[num_tables:] = x_reshape.flatten()
        return subproto

    def perturb_noise(self, proto, noise):
        # Perturb only in the deterministic case.
        if self.deterministic:
            return torch.clamp(proto + noise, 0.0, 1.0)
        return proto
