from typing import Any, Callable, TypeVar

import numpy as np
from numpy.typing import NDArray


T = TypeVar("T")
U = TypeVar("U", float, NDArray[np.float64])


def clamp(value: T, min_: Any, max_: Any) -> T:
    return min(max(value, min_), max_)


def lerp(a: U, b: U, x: Any) -> U:
    return a * (1.0 - x) + b * x


def normalize(value: U, min: Any, max: Any) -> U:
    return (value - min) / (max - min)


def quantize(value: U, step: Any, rounding_func: Callable[[Any], Any] = round) -> U:
    return rounding_func(value / step) * step


def remap_range(value: U, old_min: Any, old_max: Any, new_min: Any, new_max: Any) -> U:
    return new_min + (value - old_min) / (old_max - old_min) * (new_max - new_min)
