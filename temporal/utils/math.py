from typing import Any, Callable, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", float, NDArray[np.float_])

def lerp(a: T, b: T, x: Any) -> T:
    return a * (1.0 - x) + b * x

def normalize(value: T, min: Any, max: Any) -> T:
    return (value - min) / (max - min)

def quantize(value: T, step: Any, rounding_func: Callable[[Any], Any] = round) -> T:
    return rounding_func(value / step) * step

def remap_range(value: T, old_min: Any, old_max: Any, new_min: Any, new_max: Any) -> T:
    return new_min + (value - old_min) / (old_max - old_min) * (new_max - new_min)
