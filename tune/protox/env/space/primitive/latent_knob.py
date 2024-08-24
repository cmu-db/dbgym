from typing import Any, Optional, Sequence, Union

import gymnasium as gym
import numpy as np

from tune.protox.env.space.primitive import SettingType, is_boolean
from tune.protox.env.space.primitive.knob import CategoricalKnob, Knob, KnobMetadata


class LatentKnob(Knob):
    def __init__(
        self,
        table_name: Optional[str],
        query_name: Optional[str],
        knob_name: str,
        metadata: KnobMetadata,
        do_quantize: bool,
        default_quantize_factor: int,
        seed: int,
    ):

        super().__init__(
            table_name,
            query_name,
            knob_name,
            metadata,
            do_quantize,
            default_quantize_factor,
            seed,
        )

    def _process(self, raw_value: Any) -> Any:
        if is_boolean(self.knob_type):
            return round(raw_value)
        elif self.knob_type == SettingType.FLOAT:
            return round(raw_value, 2)
        else:
            # Consistently apply rounding.
            return int(raw_value)

    def to_latent(self, env_value: Any) -> Any:
        """Projects a point from the environment space to the network space."""
        transform_value = self.to_internal(env_value)
        # Scale into the network space.
        relative_point = (transform_value - self.space_min_value) / (
            self.space_max_value - self.space_min_value
        )
        return 2 * relative_point - 1

    def from_latent(self, latent_value: Any) -> Any:
        # This functionally assumes that the network_space and internal space maps linearly.
        # If that assumption doesn't hold, project_embedding_into_internal_space will do something wonky to the values.
        # TODO(wz2): Are there latent spaces that we don't want a linear mapping? Or we prefer a piecewise linear function?

        # First project into the [space_min_value, space_max_value] range.
        int_space = (self.space_max_value - self.space_min_value) * (
            np.round((latent_value + 1) / 2.0, 8)
        ) + self.space_min_value
        raw_value = self.to_quantize(int_space)
        if is_boolean(self.knob_type):
            return round(raw_value)
        elif self.knob_type == SettingType.FLOAT:
            return round(raw_value, 2)
        else:
            # Consistently apply rounding.
            return int(raw_value)

    def shift_offset(self, raw: Any, bin_shift: int) -> Any:
        # Specially handle the case of booleans.
        if is_boolean(self.knob_type):
            if raw == 0 and bin_shift > 0:
                return 1
            elif raw == 1 and bin_shift < 0:
                return 0
            return None

        if self.log2_scale:
            nvalue = self.to_internal(raw) + bin_shift
        else:
            nvalue = raw + self.bucket_size * bin_shift

        if nvalue < self.space_min_value or nvalue > self.space_max_value:
            # Exceeded boundaries.
            return None

        raw_value = self._process(self.to_quantize(nvalue))
        if raw_value == raw:
            # Don't return duplicates.
            return None

        return raw_value


class LatentCategoricalKnob(CategoricalKnob):
    def to_latent(self, env_value: Any) -> Any:
        return gym.spaces.utils.flatten(self, env_value)

    def from_latent(self, latent_value: Any) -> Any:
        return np.argmax(latent_value)

    def sample_weights(self, weights: Optional[Sequence[float]] = None) -> Any:
        return np.random.choice(
            [i for i in range(self.num_elems)],
            p=(
                (weights / np.sum(weights))
                if weights is not None and np.sum(weights) > 0
                else None
            ),
        )


def _create_latent_knob(
    table_name: Optional[str],
    query_name: Optional[str],
    knob_name: str,
    metadata: KnobMetadata,
    do_quantize: bool,
    default_quantize_factor: int,
    seed: int,
) -> Union[LatentKnob, LatentCategoricalKnob]:

    if "default_value" in metadata:
        return LatentCategoricalKnob(
            table_name=table_name,
            query_name=query_name,
            knob_name=knob_name,
            metadata=metadata,
            seed=seed,
        )

    return LatentKnob(
        table_name=table_name,
        query_name=query_name,
        knob_name=knob_name,
        metadata=metadata,
        do_quantize=do_quantize,
        default_quantize_factor=default_quantize_factor,
        seed=seed,
    )
