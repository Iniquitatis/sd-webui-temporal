import re
from collections.abc import Iterable, Sequence
from typing import Callable, Iterator, Optional, TypeVar


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


def find_by_predicate(iterable: Iterable[T], pred: Callable[[T], bool]) -> Optional[T]:
    for item in iterable:
        if pred(item):
            return item


def find_index_by_predicate(iterable: Iterable[T], pred: Callable[[T], bool]) -> int:
    for i, item in enumerate(iterable):
        if pred(item):
            return i

    return -1


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


def get_with_fallback(sequence: Sequence[T], index: int, fallback: U = None) -> T | U:
    return sequence[index] if index < len(sequence) else fallback


def natural_sort(iterable: Iterable[str]) -> list[str]:
    return sorted(iterable, key = lambda item: tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", item)))


def reorder_dict(d: dict[T, U], order: Iterable[T]) -> dict[T, U]:
    return {x: d[x] for x in order} | d


def swap_kv(d: dict[T, U]) -> dict[U, T]:
    return {v: k for k, v in d.items()}
