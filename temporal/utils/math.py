from collections.abc import Iterable, Sequence
from functools import reduce
from operator import mul
from typing import Any, Callable, TypeVar

from temporal.utils.numpy import FloatArray


T = TypeVar("T")
U = TypeVar("U", float, FloatArray)


def cartesian_product_at(*sets: Sequence[Any], index: int, major: bool = True) -> tuple[Any, ...]:
    result = []

    for set in reversed(sets) if major else sets:
        count = len(set)
        result.append(set[index % count])
        index //= count

    return tuple(reversed(result) if major else result)


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


def remap_range(value: U, old_min: Any, old_max: Any, new_min: Any, new_max: Any) -> U:
    return new_min + (value - old_min) / (old_max - old_min) * (new_max - new_min)
