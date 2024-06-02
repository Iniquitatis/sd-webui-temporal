import re
from copy import copy
from typing import Any, Iterator, TypeVar

from temporal.utils import logging


T = TypeVar("T")


def copy_with_overrides(obj: T, **overrides: Any) -> T:
    instance = copy(obj)

    for key, value in overrides.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
        else:
            logging.warning(f"Key {key} doesn't exist in {instance.__class__.__name__}")

    return instance


def get_with_overrides(obj: T, **overrides: Any) -> T:
    for key, value in overrides.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
        else:
            logging.warning(f"Key {key} doesn't exist in {obj.__class__.__name__}")

    return obj


def get_property_by_path(obj: Any, path: str) -> Any:
    stack = [obj]

    for part_type, part in _iter_property_path(path):
        top = stack[-1]

        if part_type == "dict_index":
            new_top = top.__getitem__(part)
        elif part_type == "list_index":
            new_top = top.__getitem__(int(part))
        else:
            new_top = getattr(top, part)

        stack.append(new_top)

    return stack[-1]


def set_property_by_path(obj: Any, path: str, value: Any) -> Any:
    stack = [obj]
    keys = []

    for part_type, part in _iter_property_path(path):
        top = stack[-1]

        if part_type == "dict_index":
            new_top = top.__getitem__(part)
        elif part_type == "list_index":
            new_top = top.__getitem__(int(part))
        else:
            new_top = getattr(top, part, None)

        stack.append(new_top)
        keys.append(part)

    obj = stack[-2]
    key = keys[-1]

    if isinstance(obj, (dict, list)):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _iter_property_path(path: str) -> Iterator[tuple[str, str]]:
    for match in re.finditer(r"\w+|\[.+?\]", path):
        key = match[0]

        if index_match := re.match(r"\[\'(.+?)\'\]", key):
            yield "dict_index", index_match[1]
        elif index_match := re.match(r"\[(.+?)\]", key):
            yield "list_index", index_match[1]
        else:
            yield "property_name", key
