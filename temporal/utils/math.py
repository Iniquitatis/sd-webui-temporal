from collections.abc import Iterable
from functools import reduce
from operator import mul
from typing import Any, Callable, TypeVar

from temporal.utils.numpy import FloatArray


T = TypeVar("T")
U = TypeVar("U", float, FloatArray)


def clamp(value: T, min_: Any, max_: Any) -> T:
    return min(max(value, min_), max_)


def lerp(a: U, b: U, x: Any) -> U:
    return a * (1.0 - x) + b * x


def normalize(value: U, min: Any, max: Any) -> U:
    return (value - min) / (max - min)


def product(iterable: Iterable[T]) -> T:
    return reduce(mul, iterable)


def quantize(value: U, step: Any, rounding_func: Callable[[Any], Any] = round) -> U:
    return rounding_func(value / step) * step


def ratio(values: Iterable[float | int]) -> tuple[float, ...]:
    if (minimum := min(values)) == 0:
        raise ValueError

    return tuple(x / minimum for x in values)


def remap_range(value: U, old_min: Any, old_max: Any, new_min: Any, new_max: Any) -> U:
    return new_min + (value - old_min) / (old_max - old_min) * (new_max - new_min)
