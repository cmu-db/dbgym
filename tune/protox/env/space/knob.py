from __future__ import annotations
from typing import Any, Sequence
import numpy as np
import gymnasium as gym
import math
from tune.protox.env import KnobClass, SettingType, is_knob_enum, resolve_enum_value
from gymnasium.spaces import Space, Box, Discrete
from gymnasium.spaces.utils import flatten, flatten_space, flatdim, unflatten


def full_knob_name(table=None, query=None, knob_name=None):
    if table is not None:
        return f"{table}_{knob_name}"
    elif query is not None:
        return f"{query}_{knob_name}"
    else:
        return knob_name


def _parse_categorical(type_str):
    if type_str == "scanmethod_enum_categorical":
        return SettingType.SCANMETHOD_ENUM_CATEGORICAL, 2
    elif type_str == "magic_hintset_enum_categorical":
        return SettingType.MAGIC_HINTSET_ENUM_CATEGORICAL, 5

    assert False, print(type_str)


def _parse_setting_dtype(type_str):
    type_str = type_str.lower()
    if type_str == "boolean":
        return SettingType.BOOLEAN, np.int32
    elif type_str == "binary_enum":
        return SettingType.BINARY_ENUM, np.int32
    elif type_str == "scanmethod_enum":
        return SettingType.SCANMETHOD_ENUM, np.int32
    elif type_str == "integer":
        return SettingType.INTEGER, np.int32
    elif type_str == "bytes":
        return SettingType.BYTES, np.int32
    elif type_str == "integer_time":
        return SettingType.INTEGER_TIME, np.int32
    else:
        assert type_str == "float"
        return SettingType.FLOAT, np.float32


def _treat_boolean(setting_type):
    return setting_type in [
        SettingType.BOOLEAN,
        SettingType.BINARY_ENUM,
        SettingType.SCANMETHOD_ENUM
    ]


class Knob(Space):
    # Type of the knob.
    knob_class: KnobClass = KnobClass.INVALID
    # Table name if table knob else None
    table_name: str = None
    # Query Name if query knob else None
    query_name: str = None
    # Knob name.
    knob_name: str = None

    # Type of the knob.
    knob_type: SettingType = SettingType.INVALID
    knob_dtype = None

    # Minimum value.
    min_value: float = None
    # Maximum value.
    max_value: float = None
    # Minimum space value.
    space_min_value: float = None
    # Maximum space value.
    space_max_value: float = None
    # Adjustment factor.
    space_correction_value: float = None

    # Knob unit.
    knob_unit: float = None
    # Quantize factor.
    quantize_factor: int = None
    # Whether to use raw log-scales.
    log2_scale: bool = False
    # Size of the bucket.
    bucket_size: int = 0
    # Whether to round or floor.
    should_round: bool = False


    def __init__(
        self,
        table_name: str,
        query_name: str,
        knob_name: str,
        metadata: dict,
        do_quantize: bool,
        default_quantize_factor: int,
        seed):

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
        self.quantize_factor = (default_quantize_factor if metadata["quantize"] == -1 else metadata["quantize"]) if do_quantize else 0
        self.log2_scale = metadata["log_scale"]
        assert not self.log2_scale or (self.log2_scale and self.quantize_factor == 0)
        self.should_round = metadata.get("round", False)

        # Setup all the metadata for the knob value.
        self.space_correction_value = 0
        self.space_min_value = self.min_value = metadata["min"]
        self.space_max_value = self.max_value = metadata["max"]
        if self.log2_scale:
            self.space_correction_value = (1 - self.space_min_value)
            self.space_min_value += self.space_correction_value
            self.space_max_value += self.space_correction_value

            self.space_min_value = math.floor(math.log2(self.space_min_value))
            self.space_max_value = math.ceil(math.log2(self.space_max_value))
        elif self.quantize_factor > 0:
            if self.knob_type == SettingType.FLOAT:
                self.bucket_size = (self.max_value - self.min_value) / self.quantize_factor
            else:
                max_buckets = min(self.max_value - self.min_value, self.quantize_factor)
                self.bucket_size = (self.max_value - self.min_value) / max_buckets

        super().__init__((), self.knob_dtype, seed=seed)

    def name(self):
        # Construct the name.
        return full_knob_name(self.table_name, self.query_name, self.knob_name)

    def invert(self, val):
        if _treat_boolean(self.knob_type):
            if val == 1.:
                return 0.
            else:
                return 1.
        return val

    def _log2_quantize_to_internal(self, transform_value: Any):
        # Perform the correct log2 quantization.
        assert self.log2_scale
        transform_value += self.space_correction_value
        transform_value = math.log2(transform_value)
        return transform_value

    def _project_embedding_into_internal_space(self, value: Any):
        network_space = (-1., 1.)
        # Assume that network space and internal space obey a simple scale mapping.
        # First project into the [space_min_value, space_max_value] range.
        relative_old_space = np.round((value + 1) / 2., 8)
        new_space = (self.space_max_value - self.space_min_value) * relative_old_space + self.space_min_value
        # Internal space might differ if scaling and/or other transformations are done.
        return new_space

    def _quantize_internal_to_raw(self, raw_value: Any):
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
        if self.should_round:
            quantized_value = round(raw_value / self.bucket_size) * self.bucket_size
        else:
            quantized_value = math.floor(raw_value / self.bucket_size) * self.bucket_size
        return np.clip(quantized_value, self.min_value, self.max_value)

    def adjust_quantize_bin(self, value: Any, bin_shift: int):
        # Specially handle the case of booleans.
        if _treat_boolean(self.knob_type):
            if value == 0 and bin_shift > 0:
                return 1
            elif value == 1 and bin_shift < 0:
                return 0
            return None

        new_internal_value = None
        if self.log2_scale:
            # Adjust the log2 integral bin if we are in log2 scaling.
            log2_bin = self._log2_quantize_to_internal(value)
            new_bin = log2_bin + bin_shift
            # We use the space_min_value because that is the bounds on the log scale.
            if new_bin >= self.space_min_value and new_bin <= self.space_max_value and new_bin != log2_bin:
                new_internal_value = new_bin
        else:
            # Adjust the raw value itself if we don't use log2 scaling.
            target_value = value + self.bucket_size * bin_shift
            if target_value >= self.min_value and target_value <= self.max_value and target_value != value:
                new_internal_value = target_value

        if new_internal_value is None:
            return None

        raw_value = self._quantize_internal_to_raw(new_internal_value)
        if _treat_boolean(self.knob_type):
            return round(raw_value)
        elif self.knob_type == SettingType.FLOAT:
            return round(raw_value, 2)
        else:
            # Consistently apply rounding.
            return int(raw_value)

    def project_env_to_embedding(self, value: Any):
        """Projects a point from the environment space to the network space."""
        if self.log2_scale:
            # First apply the log2_scale if necessary.
            transform_value = self._log2_quantize_to_internal(value)
        else:
            transform_value = value

        # Scale into the network space.
        network_space = (-1., 1.)
        relative_point = (transform_value - self.space_min_value) / (self.space_max_value - self.space_min_value)
        network_space = relative_point * (network_space[1] - network_space[0]) + network_space[0]
        return network_space

    def project_embedding_to_env(self, value: Any):
        """Projects a point from network to the environment space."""
        # This functionally assumes that the network_space and internal space maps linearly.
        # If that assumption doesn't hold, project_embedding_into_internal_space will do something wonky to the values.
        raw_value = self._quantize_internal_to_raw(self._project_embedding_into_internal_space(value))
        if _treat_boolean(self.knob_type):
            return round(raw_value)
        elif self.knob_type == SettingType.FLOAT:
            return round(raw_value, 2)
        else:
            # Consistently apply rounding.
            return int(raw_value)

    def project_scraped_setting(self, value: Any):
        """Projects a point from the DBMS into the (possibly) more constrained environment space."""
        # Constrain the value to be within the actual min/max range.
        value = np.clip(value, self.min_value, self.max_value)
        return self.project_embedding_to_env(self.project_env_to_embedding(value))

    def resolve_per_query_knob(self, val, all_knobs={}):
        assert self.knob_class == KnobClass.QUERY
        if is_knob_enum(self):
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
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return x >= self.min_value and x <= self.max_value

    def to_jsonable(self, sample_n: Sequence[Any]) -> list[Any]:
        """Convert a batch of samples from this space to a JSONable data type."""
        return [sample for sample in sample_n]

    def from_jsonable(self, sample_n: Sequence[float | int]) -> list[Any]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return np.array(sample_n).astype(self.dtype)

    def sample(self, mask: None = None) -> Any:
        """Samples a point from the environment action space subject to action space constraints."""
        assert False

    def is_categorical(self):
        return False


@flatten.register(Knob)
def _flatten_knob(space: Knob, x: Any) -> NDArray[Any]:
    return [x]


@unflatten.register(Knob)
def _unflatten_knob(space: Knob, x: NDArray[Any]) -> Any:
    return x[0]


@flatten_space.register(Knob)
def _flatten_space_knob(space: Knob) -> Box:
    return Box(low=space.space_min_value, high=space.space_max_value, shape=(1,), dtype=space.knob_dtype)


@flatdim.register(Knob)
def _flatdim_knob(space: Knob) -> int:
    return 1


class CategoricalKnob(Discrete):
    # Type of the knob.
    knob_class: KnobClass = KnobClass.INVALID
    # Table name if table knob else None
    table_name: str = None
    # Query Name if query knob else None
    query_name: str = None
    # Knob name.
    knob_name: str = None

    # Type of the knob.
    knob_type: SettingType = SettingType.INVALID
    num_elems = 0
    default_value = 0
    values = []

    def is_categorical(self):
        return True

    def invert(self, val):
        return val

    def __init__(
        self,
        table_name: str,
        query_name: str,
        knob_name: str,
        metadata: dict,
        seed):

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
            self.knob_type, self.num_elems = _parse_categorical(metadata["type"])
        self.default_value = metadata["default"]
        super().__init__(self.num_elems, seed=seed)

    def sample_uniform(self):
        return np.random.randint(0, self.num_elems)

    def name(self):
        # Construct the name.
        return full_knob_name(self.table_name, self.query_name, self.knob_name)

    def project_env_to_embedding(self, value: Any):
        """Projects a point from the environment space to the network space."""
        return gym.spaces.utils.flatten(self, value)

    def project_embedding_to_env(self, value: Any):
        """Projects a point from network to the environment space."""
        return np.argmax(value)

    def project_scraped_setting(self, value: Any):
        """Projects a point from the DBMS into the (possibly) more constrained environment space."""
        # Constrain the value to be within the actual min/max range.
        assert False

    def resolve_per_query_knob(self, val, all_knobs={}):
        assert self.knob_class == KnobClass.QUERY
        assert is_knob_enum(self)
        return resolve_enum_value(self, val, all_knobs=all_knobs)

    def sample(self, mask: None = None) -> Any:
        """Samples a point from the environment action space subject to action space constraints."""
        assert False

    def _sample_weights(self, weights) -> Any:
        return np.random.choice(
            [i for i in range(self.num_elems)],
            p=(weights / np.sum(weights)) if np.sum(weights) > 0 else None)


def _create_knob(
    table_name: str,
    query_name: str,
    knob_name: str,
    metadata: dict,
    do_quantize: bool,
    default_quantize_factor: int,
    seed):

    if "default" in metadata:
        return CategoricalKnob(
            table_name=table_name,
            query_name=query_name,
            knob_name=knob_name,
            metadata=metadata,
            seed=seed)

    return Knob(
        table_name=table_name,
        query_name=query_name,
        knob_name=knob_name,
        metadata=metadata,
        do_quantize=do_quantize,
        default_quantize_factor=default_quantize_factor,
        seed=seed)
