import math
from typing import Any, Optional, Sequence, Tuple, TypedDict, Union, cast

import numpy as np
from gymnasium.spaces import Box, Discrete, Space
from gymnasium.spaces.utils import flatdim, flatten, flatten_space, unflatten
from numpy.typing import NDArray

from tune.protox.env.space.primitive import (
    KnobClass,
    SettingType,
    is_boolean,
    is_knob_enum,
)


def full_knob_name(
    table: Optional[str] = None, query: Optional[str] = None, knob_name: str = ""
) -> str:
    assert knob_name != ""

    if table is not None:
        return f"{table}_{knob_name}"
    elif query is not None:
        return f"{query}_{knob_name}"
    else:
        return knob_name


def _parse_setting_dtype(type_str: str) -> Tuple[SettingType, Any]:
    return {
        "boolean": (SettingType.BOOLEAN, np.int32),
        "integer": (SettingType.INTEGER, np.int32),
        "bytes": (SettingType.BYTES, np.int32),
        "integer_time": (SettingType.INTEGER_TIME, np.int32),
        "float": (SettingType.FLOAT, np.float32),
        "binary_enum": (SettingType.BINARY_ENUM, np.int32),
        "scanmethod_enum": (SettingType.SCANMETHOD_ENUM, np.int32),
        "query_table_enum": (SettingType.QUERY_TABLE_ENUM, np.int32),
    }[type_str]


class KnobMetadata(TypedDict, total=False):
    type: str
    min: float
    max: float
    quantize: bool
    log_scale: int
    unit: int
    values: list[str]
    default_value: int


class Knob(Space[Any]):
    def __init__(
        self,
        table_name: Optional[str],
        query_name: Optional[str],
        knob_name: str,
        metadata: KnobMetadata,
        do_quantize: bool,
        default_quantize_factor: int,
        seed: int,
    ) -> None:

        self.table_name = table_name
        self.query_name = query_name
        self.knob_name = knob_name
        if table_name is not None:
            self.knob_class = KnobClass.TABLE
        elif query_name is not None:
            self.knob_class = KnobClass.QUERY
        else:
            self.knob_class = KnobClass.KNOB

        self.knob_type, self.knob_dtype = _parse_setting_dtype(metadata["type"])
        self.knob_unit = metadata["unit"]
        self.quantize_factor = (
            (
                default_quantize_factor
                if metadata["quantize"] == -1
                else metadata["quantize"]
            )
            if do_quantize
            else 0
        )
        self.log2_scale = metadata["log_scale"]
        assert not self.log2_scale or (self.log2_scale and self.quantize_factor == 0)

        # Setup all the metadata for the knob value.
        self.space_correction_value = 0.0
        self.space_min_value = self.min_value = metadata["min"]
        self.space_max_value = self.max_value = metadata["max"]
        self.bucket_size = 0.0
        if self.log2_scale:
            self.space_correction_value = 1.0 - self.space_min_value
            self.space_min_value += self.space_correction_value
            self.space_max_value += self.space_correction_value

            self.space_min_value = math.floor(math.log2(self.space_min_value))
            self.space_max_value = math.ceil(math.log2(self.space_max_value))
        elif self.quantize_factor > 0:
            if self.knob_type == SettingType.FLOAT:
                self.bucket_size = (
                    self.max_value - self.min_value
                ) / self.quantize_factor
            else:
                max_buckets = min(self.max_value - self.min_value, self.quantize_factor)
                self.bucket_size = (self.max_value - self.min_value) / max_buckets

        super().__init__((), self.knob_dtype, seed=seed)

    def name(self) -> str:
        # Construct the name.
        return full_knob_name(self.table_name, self.query_name, self.knob_name)

    def to_internal(self, env_value: Any) -> Any:
        if self.log2_scale:
            return math.log2(env_value + self.space_correction_value)
        return env_value

    def to_quantize(self, raw_value: Any) -> Any:
        """Adjusts the raw value to the quantized bin value."""
        assert raw_value >= self.space_min_value and raw_value <= self.space_max_value

        # Handle log scaling values.
        if self.log2_scale:
            # We integralize the log-space to exploit the log-scaling and discretization.
            proj_value = pow(2, round(raw_value))
            # Possibly adjust with the correction bias now.
            proj_value -= self.space_correction_value
            return np.clip(proj_value, self.min_value, self.max_value)

        # If we don't quantize, don't quantize.
        if self.quantize_factor is None or self.quantize_factor == 0:
            return np.clip(raw_value, self.min_value, self.max_value)

        # FIXME: We currently basically bias aggressively against the lower bucket, under the prior
        # belief that the incremental gain of going higher is less potentially / more consumption
        # and so it is ok to bias lower.
        quantized_value = (
            math.floor(round(raw_value / self.bucket_size, 8)) * self.bucket_size
        )
        return np.clip(quantized_value, self.min_value, self.max_value)

    def project_scraped_setting(self, value: Any) -> Any:
        """Projects a point from the DBMS into the (possibly) more constrained environment space."""
        # Constrain the value to be within the actual min/max range.
        value = np.clip(value, self.min_value, self.max_value)
        return self.to_quantize(self.to_internal(value))

    def resolve_per_query_knob(self, val: Any, all_knobs: dict[str, Any] = {}) -> str:
        assert self.knob_class == KnobClass.QUERY
        if is_knob_enum(self.knob_type):
            return resolve_enum_value(self, val, all_knobs=all_knobs)
        else:
            kt = self.knob_type
            if kt == SettingType.FLOAT:
                param = f"{val:.2f}"
            elif kt == SettingType.BOOLEAN:
                param = "on" if val == 1 else "off"
            else:
                param = f"{val:d}"

            return f"Set ({self.knob_name} {param})"

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return cast(bool, x >= self.min_value and x <= self.max_value)

    def to_jsonable(self, sample_n: Sequence[Any]) -> list[Any]:
        """Convert a batch of samples from this space to a JSONable data type."""
        return [sample for sample in sample_n]

    def from_jsonable(self, sample_n: Sequence[Union[float, int]]) -> Any:
        """Convert a JSONable data type to a batch of samples from this space."""
        return np.array(sample_n).astype(self.dtype)

    def sample(self, mask: None = None) -> Any:
        """Samples a point from the environment action space subject to action space constraints."""
        raise NotImplementedError()

    def invert(self, value: Any) -> Any:
        if is_boolean(self.knob_type):
            if value == 1:
                return 0
            else:
                return 1
        return value

    def __hash__(self) -> int:
        return hash(self.name())

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            assert isinstance(other, Knob)
            return self.name() == other.name()

        return False


@flatten.register(Knob)
def _flatten_knob(space: Knob, x: Any) -> NDArray[Any]:
    return np.array([x], np.float32)


@unflatten.register(Knob)
def _unflatten_knob(space: Knob, x: NDArray[Any]) -> Any:
    return x[0]


@flatten_space.register(Knob)
def _flatten_space_knob(space: Knob) -> Box:
    return Box(
        low=space.space_min_value,
        high=space.space_max_value,
        shape=(1,),
        dtype=space.knob_dtype,
    )


@flatdim.register(Knob)
def _flatdim_knob(space: Knob) -> int:
    return 1


def _categorical_elems(type_str: str) -> Tuple[SettingType, int]:
    return {
        "scanmethod_enum_categorical": (SettingType.SCANMETHOD_ENUM_CATEGORICAL, 2),
    }[type_str]


class CategoricalKnob(Discrete):
    def __init__(
        self,
        table_name: Optional[str],
        query_name: Optional[str],
        knob_name: str,
        metadata: KnobMetadata,
        seed: int,
    ) -> None:

        self.table_name = table_name
        self.query_name = query_name
        self.knob_name = knob_name
        assert self.table_name is None and self.query_name is not None
        self.knob_class = KnobClass.QUERY

        if metadata["type"] == "query_table_enum":
            self.knob_type = SettingType.QUERY_TABLE_ENUM
            self.num_elems = len(metadata["values"]) + 1
            self.values = metadata["values"]
        else:
            self.knob_type, self.num_elems = _categorical_elems(metadata["type"])
        self.default_value = metadata["default_value"]
        super().__init__(self.num_elems, seed=seed)

    def name(self) -> str:
        # Construct the name.
        return full_knob_name(self.table_name, self.query_name, self.knob_name)

    def project_scraped_setting(self, value: Any) -> Any:
        """Projects a point from the DBMS into the (possibly) more constrained environment space."""
        # Constrain the value to be within the actual min/max range.
        raise NotImplementedError()

    def sample(self, mask: Optional[NDArray[np.int8]] = None) -> Any:
        """Samples a point from the environment action space subject to action space constraints."""
        return np.random.randint(0, self.num_elems)

    def resolve_per_query_knob(self, val: Any, all_knobs: dict[str, Any] = {}) -> str:
        assert self.knob_class == KnobClass.QUERY
        assert is_knob_enum(self.knob_type)
        return resolve_enum_value(self, val, all_knobs=all_knobs)

    def invert(self, value: Any) -> Any:
        return value

    def __hash__(self) -> int:
        return hash(self.name())

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            assert isinstance(other, CategoricalKnob)
            return self.name() == other.name()

        return False


def resolve_enum_value(
    knob: Union[Knob, CategoricalKnob], value: Any, all_knobs: dict[str, Any] = {}
) -> str:
    assert is_knob_enum(knob.knob_type)
    if knob.knob_type == SettingType.BINARY_ENUM:
        return "on" if value == 1 else "off"

    if knob.knob_type == SettingType.QUERY_TABLE_ENUM:
        assert isinstance(knob, CategoricalKnob)
        integral_value = int(value)
        if integral_value == 0:
            return ""

        assert "max_worker_processes" in all_knobs
        max_workers = all_knobs["max_worker_processes"]

        selected_table = knob.values[integral_value - 1]
        # FIXME: pg_hint_plan lets specifying any and then pg will tweak it down.
        return f"Parallel({selected_table} {max_workers})"

    if knob.knob_type in [
        SettingType.SCANMETHOD_ENUM,
        SettingType.SCANMETHOD_ENUM_CATEGORICAL,
    ]:
        assert "_scanmethod" in knob.knob_name
        tbl = knob.knob_name.split("_scanmethod")[0]
        if value == 1:
            return f"IndexOnlyScan({tbl})"
        return f"SeqScan({tbl})"

    raise ValueError(f"Unsupported knob num {knob.knob_type}")


def _create_knob(
    table_name: Optional[str],
    query_name: Optional[str],
    knob_name: str,
    metadata: KnobMetadata,
    do_quantize: bool,
    default_quantize_factor: int,
    seed: int,
) -> Union[Knob, CategoricalKnob]:

    if "default_value" in metadata:
        return CategoricalKnob(
            table_name=table_name,
            query_name=query_name,
            knob_name=knob_name,
            metadata=metadata,
            seed=seed,
        )

    return Knob(
        table_name=table_name,
        query_name=query_name,
        knob_name=knob_name,
        metadata=metadata,
        do_quantize=do_quantize,
        default_quantize_factor=default_quantize_factor,
        seed=seed,
    )
