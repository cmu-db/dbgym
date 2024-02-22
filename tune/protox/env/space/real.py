"""Implementation of a space that represents closed boxes in euclidean space."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, SupportsFloat

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium.spaces.space import Space
from gymnasium.spaces.utils import flatdim
from gymnasium.spaces.utils import flatten
from gymnasium.spaces.utils import unflatten


class Real(Space[Any]):
    def __init__(
        self,
        low: SupportsFloat | NDArray[Any],
        high: SupportsFloat | NDArray[Any],
        dtype: type[np.floating[Any]] | type[np.integer[Any]] = np.float32,
        seed: int | np.random.Generator | None = None,
    ):
        assert (
            dtype is not None
        ), "Box dtype must be explicitly provided, cannot be None."
        self.dtype = np.dtype(dtype)

        self.low = low
        self.high = high
        super().__init__((1,), self.dtype, seed)

    @property
    def shape(self) -> tuple[int, ...]:
        """Has stricter type than gym.Space - never None."""
        return 1

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True

    def sample(self, mask: None = None) -> NDArray[Any]:
        assert False

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return bool(x >= self.low and x <= self.high)

    def to_jsonable(self, sample_n: Sequence[NDArray[Any]]) -> list[list]:
        """Convert a batch of samples from this space to a JSONable data type."""
        return [sample_n]

    def from_jsonable(self, sample_n: Sequence[float | int]) -> list[NDArray[Any]]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [sample_n]

    def __repr__(self) -> str:
        return f"Real({self.low}, {self.high}, {self.dtype})"

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype equivalence."""
        return (
            isinstance(other, Real)
            # and (self.dtype == other.dtype)
            and self.low == other.low
            and self.high == other.high
        )

@flatdim.register(Real)
def _flatdim_real(space: Real) -> int:
    return 1

@flatten.register(Real)
def _flatten_real(space: Real, x: Any) -> Any:
    return [x]

@unflatten.register(Real)
def _unflatten_real(space: Real, x: Any) -> Any:
    return x.astype(space.dtype)
