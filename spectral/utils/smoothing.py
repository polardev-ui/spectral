from collections import deque
from typing import List, Optional
import math


class ExponentialSmoothing:
    def __init__(self, alpha: float = 0.5):
        self._alpha = max(0.0, min(1.0, alpha))
        self._value = None

    def update(self, new_value: float) -> float:
        if self._value is None:
            self._value = new_value
        else:
            self._value = self._alpha * self._value + (1 - self._alpha) * new_value
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value

    def reset(self):
        self._value = None


class OneEuroFilter:
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def _exponential_smoothing(
        self, a: float, x: float, x_prev: float
    ) -> float:
        return a * x + (1 - a) * x_prev

    def update(self, x: float, t: float) -> float:
        if self._t_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x

        t_e = t - self._t_prev
        if t_e <= 0:
            return self._x_prev

        a_d = self._smoothing_factor(t_e, self._d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self._dx_prev)

        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self._x_prev)

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t

        return x_hat

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class MovingAverage:
    def __init__(self, window_size: int = 5):
        self._window_size = window_size
        self._buffer = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        self._buffer.append(value)
        return sum(self._buffer) / len(self._buffer)

    @property
    def value(self) -> Optional[float]:
        if not self._buffer:
            return None
        return sum(self._buffer) / len(self._buffer)

    def reset(self):
        self._buffer.clear()


class WeightedMovingAverage:
    def __init__(self, window_size: int = 5):
        self._window_size = window_size
        self._buffer = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        self._buffer.append(value)
        weights = list(range(1, len(self._buffer) + 1))
        total_weight = sum(weights)
        weighted_sum = sum(v * w for v, w in zip(self._buffer, weights))
        return weighted_sum / total_weight

    def reset(self):
        self._buffer.clear()


class VelocityTracker:
    def __init__(self, window_size: int = 5):
        self._window_size = window_size
        self._positions = deque(maxlen=window_size)
        self._timestamps = deque(maxlen=window_size)

    def update(self, position: float, timestamp: float) -> float:
        self._positions.append(position)
        self._timestamps.append(timestamp)

        if len(self._positions) < 2:
            return 0.0

        dt = self._timestamps[-1] - self._timestamps[0]
        if dt < 1e-6:
            return 0.0

        dp = self._positions[-1] - self._positions[0]
        return dp / dt

    @property
    def velocity(self) -> float:
        if len(self._positions) < 2:
            return 0.0
        dt = self._timestamps[-1] - self._timestamps[0]
        if dt < 1e-6:
            return 0.0
        dp = self._positions[-1] - self._positions[0]
        return dp / dt

    def reset(self):
        self._positions.clear()
        self._timestamps.clear()
