import re
from collections.abc import Iterable
from typing import Iterator, TypeVar


T = TypeVar("T")
U = TypeVar("U")


def batched(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    batch = []

    for item in iterable:
        batch.append(item)

        if len(batch) < size:
            continue

        yield batch

        batch.clear()

    if len(batch) > 0:
        yield batch


def get_first_element(iterable: Iterable[T], fallback: U = None) -> T | U:
    return next(iter(iterable)) if iterable else fallback


def get_next_element(iterable: Iterable[T], current: T, fallback: U = None) -> T | U:
    iterator = iter(iterable)

    while True:
        try:
            item = next(iterator)
        except StopIteration:
            return fallback

        if item == current:
            break

    try:
        return next(iterator)
    except StopIteration:
        return fallback


def natural_sort(iterable: Iterable[str]) -> list[str]:
    return sorted(iterable, key = lambda item: tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", item)))


def reorder_dict(d: dict[T, U], order: Iterable[T]) -> dict[T, U]:
    return {x: d[x] for x in order} | d
