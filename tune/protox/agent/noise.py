from abc import ABC, abstractmethod
from typing import Optional
from numpy.typing import NDArray
import numpy as np


class ActionNoise(ABC):
    """
    The action noise base class
    """

    def __init__(self, mean: NDArray[np.float32], sigma: NDArray[np.float32]) -> None:
        super().__init__()
        self._mu = mean
        self._sigma = sigma

    def reset(self) -> None:
        """
        Call end of episode reset for the noise
        """
        pass

    @abstractmethod
    def __call__(self) -> NDArray[np.float32]:
        raise NotImplementedError()


class NormalActionNoise(ActionNoise):
    """
    A Gaussian action noise.

    :param mean: Mean value of the noise
    :param sigma: Scale of the noise (std here)
    :param dtype: Type of the output noise
    """

    def __init__(
        self,
        mean: NDArray[np.float32],
        sigma: NDArray[np.float32],
    ) -> None:
        super().__init__(mean, sigma)

    def __call__(self) -> NDArray[np.float32]:
        return np.random.normal(self._mu, self._sigma).astype(np.float32)

    def __repr__(self) -> str:
        return f"NormalActionNoise(mu={self._mu}, sigma={self._sigma})"


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    An Ornstein Uhlenbeck action noise, this is designed to approximate Brownian motion with friction.

    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    :param mean: Mean of the noise
    :param sigma: Scale of the noise
    :param theta: Rate of mean reversion
    :param dt: Timestep for the noise
    :param initial_noise: Initial value for the noise output, (if None: 0)
    :param dtype: Type of the output noise
    """

    def __init__(
        self,
        mean: NDArray[np.float32],
        sigma: NDArray[np.float32],
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_noise: Optional[NDArray[np.float32]] = None,
    ) -> None:
        super().__init__(mean, sigma)
        self._theta = theta
        self._dt = dt
        self.initial_noise = initial_noise
        self.noise_prev = np.zeros_like(self._mu)
        self.reset()

    def __call__(self) -> NDArray[np.float32]:
        noise: NDArray[np.float32] = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        )
        self.noise_prev = noise
        return noise.astype(np.float32)

    def reset(self) -> None:
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = (
            self.initial_noise
            if self.initial_noise is not None
            else np.zeros_like(self._mu)
        )

    def __repr__(self) -> str:
        return f"OrnsteinUhlenbeckActionNoise(mu={self._mu}, sigma={self._sigma})"


class ClampNoise(ActionNoise):
    def __init__(self, other: ActionNoise, clamp: float):
        super().__init__(np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))
        self.other = other
        self.other.reset()
        self.clamp = clamp

    def __call__(self) -> NDArray[np.float32]:
        return np.clip(self.other(), -self.clamp, self.clamp).astype(np.float32)

    def reset(self) -> None:
        self.other.reset()

    def __repr__(self) -> str:
        return f"ClampNoise({self.clamp}, {self.other})"