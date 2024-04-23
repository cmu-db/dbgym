from typing import ClassVar, Optional, Type, TypeVar

from tune.protox.env.types import IndexSpaceRawSample


class IndexAction(object):
    IA = TypeVar("IA", bound="IndexAction")

    index_name_counter = 0
    index_name_map: dict["IndexAction", int] = dict()

    def __init__(
        self,
        idx_type: str,
        tbl: str,
        columns: list[str],
        col_idxs: Optional[list[int]],
        inc_names: list[str],
        raw_repr: Optional[IndexSpaceRawSample],
        bias: float = 0.0,
    ) -> None:

        self.idx_type = idx_type
        self.tbl_name = tbl
        self.columns = columns
        self.col_idxs = col_idxs
        self.inc_names = inc_names
        self.raw_repr = raw_repr
        self.bias = bias

    @property
    def is_valid(self) -> bool:
        return (
            self.tbl_name is not None
            and self.columns is not None
            and len(self.columns) > 0
        )

    @classmethod
    def construct_md(
        cls: Type[IA],
        idx_name: str,
        table: str,
        idx_type: str,
        columns: list[str],
        inc_names: list[str],
    ) -> IA:
        ia = cls(
            idx_type=idx_type,
            tbl=table,
            columns=columns,
            col_idxs=None,
            inc_names=inc_names,
            raw_repr=None,
            bias=0.0,
        )
        assert ia.get_index_name() == idx_name, f"ia.get_index_name()={ia.get_index_name()} but idx_name={idx_name}"
        return ia

    def sql(self, add: bool, allow_fail: bool = False) -> str:
        idx_name = self.get_index_name()
        if not add:
            if allow_fail:
                return f"DROP INDEX IF EXISTS {idx_name}"
            return f"DROP INDEX {idx_name}"

        return "CREATE INDEX {flag} {idx_name} ON {tbl_name} USING {idx_type} ({columns}) {inc_clause}".format(
            flag="IF NOT EXISTS" if allow_fail else "",
            idx_name=idx_name,
            tbl_name=self.tbl_name,
            idx_type=self.idx_type,
            columns=",".join(self.columns),
            inc_clause=(
                ""
                if len(self.inc_names) == 0
                else "INCLUDE (" + ",".join(self.inc_names) + ")"
            ),
        )

    # A given index name (like "index5") maps one-to-one to the function of an
    # index (i.e. its table, columns, etc.).
    def get_index_name(self):
        if self not in IndexAction.index_name_map:
            IndexAction.index_name_map[self] = f"index{IndexAction.index_name_counter}"
            IndexAction.index_name_counter += 1
        
        return IndexAction.index_name_map[self]

    # This equality/hash mechanism is purely based off of index identity.
    # We ensure that all other flags are exclusive from a "validity" pre-check.
    #
    # For instance, when de-duplication, one needs to check that the IndexAction
    # can *actually* be used before relying on the identity test. Can't drop an
    # index that doesn't exist; can't create an index that does for instance.
    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            assert isinstance(other, IndexAction)
            ts = set(self.inc_names)
            os = set(other.inc_names)
            is_eq = (
                self.idx_type == other.idx_type
                and self.tbl_name == other.tbl_name
                and self.columns == other.columns
                and ts == os
            )
            return is_eq
        return False

    def __hash__(self) -> int:
        h = hash(
            (
                self.idx_type,
                self.tbl_name,
                tuple(self.columns),
                tuple(sorted(set(self.inc_names))),
            )
        )
        return h

    def __repr__(self) -> str:
        return "CREATE {idx_name} ON {tbl_name} USING {idx_type} ({columns}) {inc_clause}".format(
            idx_name=self.get_index_name(),
            tbl_name=self.tbl_name,
            idx_type=self.idx_type,
            columns=",".join(self.columns),
            inc_clause=(
                ""
                if len(self.inc_names) == 0
                else "INCLUDE (" + ",".join(self.inc_names) + ")"
            ),
        )
