from enum import Enum, unique
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    NewType,
    Optional,
    Tuple,
    TypeAlias,
    TypedDict,
    Union,
)

import torch

from tune.protox.env.space.primitive.knob import CategoricalKnob, Knob

# https://mypy.readthedocs.io/en/stable/common_issues.html#import-cycles
if TYPE_CHECKING:
    from tune.protox.env.space.primitive.index import IndexAction


@unique
class QueryType(Enum):
    UNKNOWN = -1
    SELECT = 0
    CREATE_VIEW = 1
    DROP_VIEW = 2
    INS_UPD_DEL = 3


class NeighborParameters(TypedDict, total=False):
    knob_num_nearest: int
    knob_span: int
    index_num_samples: int
    index_rules: bool


class ServerIndexMetadata(TypedDict, total=False):
    index_type: str
    columns: list[str]
    include: list[str]


DEFAULT_NEIGHBOR_PARAMETERS = NeighborParameters(
    {
        "knob_num_nearest": 100,
        "knob_span": 1,
        "index_num_samples": 100,
        "index_rules": True,
    }
)

# {table: {index1: ServerIndexMetadata, index2: ...}, ...}
ServerTableIndexMetadata = NewType(
    "ServerTableIndexMetadata", dict[str, dict[str, ServerIndexMetadata]]
)
ProtoAction = NewType("ProtoAction", torch.Tensor)  # type: ignore

KnobMap = NewType("KnobMap", dict[str, Union[Knob, CategoricalKnob]])
KnobSpaceRawAction = NewType("KnobSpaceRawAction", torch.Tensor)  # type: ignore
# {knob.name(): knob_value, ...}
KnobSpaceAction = NewType("KnobSpaceAction", dict[str, Any])
# {knob.name(): knob_value, ...}
KnobSpaceContainer = NewType("KnobSpaceContainer", dict[str, Any])

# {KnobObject: knob_value, ...}
QuerySpaceKnobAction = NewType(
    "QuerySpaceKnobAction", dict[Union[Knob, CategoricalKnob], Any]
)
# {knob.name(): knob_value, ...}
QuerySpaceAction: TypeAlias = KnobSpaceAction
# {knob.name(): knob_value, ...}
QuerySpaceContainer: TypeAlias = KnobSpaceContainer

# ([idx_type], [table_encoding], [key1_encoding], ... [key#_encoding], [include_mask])
IndexSpaceRawSample = NewType("IndexSpaceRawSample", Tuple[Any, ...])
# [IndexAction(index1), ...]
IndexSpaceContainer = NewType("IndexSpaceContainer", list["IndexAction"])

# (table_name, column_name)
TableColTuple = NewType("TableColTuple", Tuple[str, str])

# {table: [att1, att2, ...], ...}
TableAttrListMap = NewType("TableAttrListMap", dict[str, list[str]])
TableAttrSetMap = NewType("TableAttrSetMap", dict[str, set[str]])
# {attr: [tab1, tab2, ....], ...}
AttrTableListMap = NewType("AttrTableListMap", dict[str, list[str]])

# {table: set[ (att1, att3), (att3, att4), ... ], ...}
# This maps a table to a set of attributes accessed together.
TableAttrAccessSetsMap = NewType(
    "TableAttrAccessSetsMap", dict[str, set[Tuple[str, ...]]]
)

# {qid: {table: scan_method, ...}, ...}
QueryTableAccessMap = NewType("QueryTableAccessMap", dict[str, dict[str, str]])
# {table: [alias1, alias2, ...], ...}
TableAliasMap = NewType("TableAliasMap", dict[str, list[str]])
# {qid: {table: [alias1, alias2, ...], ...}, ...}
QueryTableAliasMap = NewType("QueryTableAliasMap", dict[str, TableAliasMap])
# {qid: [(query_type1, query_str1), (query_type2, query_str2), ...], ...}
QueryMap = NewType("QueryMap", dict[str, list[Tuple[QueryType, str]]])

HolonAction = NewType(
    "HolonAction",
    Tuple[
        KnobSpaceAction,
        IndexSpaceRawSample,
        QuerySpaceAction,
    ],
)

HolonStateContainer = NewType(
    "HolonStateContainer",
    Tuple[
        KnobSpaceContainer,
        IndexSpaceContainer,
        QuerySpaceContainer,
    ],
)
HolonSubAction = Union[KnobSpaceAction, IndexSpaceRawSample, QuerySpaceAction]

QueryRun = NamedTuple(
    "QueryRun",
    [
        ("prefix", str),
        ("prefix_qid", str),
        ("qknobs", QuerySpaceKnobAction),
    ],
)

BestQueryRun = NamedTuple(
    "BestQueryRun",
    [
        ("query_run", Optional[QueryRun]),
        ("runtime", Optional[float]),
        ("timed_out", bool),
        ("explain_data", Optional[Any]),
        ("metric_data", Optional[dict[str, Any]]),
    ],
)


class TargetResetConfig(TypedDict, total=False):
    metric: Optional[float]
    env_state: Any
    config: HolonStateContainer
    query_metric_data: dict[str, BestQueryRun]


class QuerySpec(TypedDict, total=False):
    benchbase: bool
    oltp_workload: bool
    query_transactional: Union[str, Path]
    query_directory: Union[str, Path]
    query_order: Union[str, Path]

    execute_query_directory: Union[str, Path]
    execute_query_order: Union[str, Path]

    tbl_include_subsets_prune: bool
    tbl_fold_subsets: bool
    tbl_fold_delta: int
    tbl_fold_iterations: int


class EnvInfoDict(TypedDict, total=False):
    # Original baseline metric.
    baseline_metric: float
    # Original baseline reward.
    baseline_reward: float
    # Data generated from each run.
    best_query_run_data: dict[str, BestQueryRun]
    # Path to run artifacts.
    results_dpath: Optional[Union[str, Path]]

    # Previous state container.
    prior_state_container: Optional[HolonStateContainer]
    # Previous pg conf.
    prior_pgconf: Optional[Union[str, Path]]

    # Changes made to the DBMS during this step.
    attempted_changes: Tuple[list[str], list[str]]

    # Metric of this step.
    metric: float
    # Reward of this step.
    reward: float
    # Whether any queries timed out or the workload as a whole timed out.
    did_anything_time_out: bool
    # Query metric data.
    query_metric_data: Optional[dict[str, BestQueryRun]]
    # Information about the actions that were executed this step.
    # The actions are in a format usable by replay. (TODO(phw2))
    actions_info: Tuple["KnobSpaceAction", "IndexAction", "QuerySpaceAction"]
    # ProtoAction of the altered step action.
    maximal_embed: ProtoAction

    # New state container.
    state_container: HolonStateContainer
    # What the LSC associated with the action is.
    lsc: float
