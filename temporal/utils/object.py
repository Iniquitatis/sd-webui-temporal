from copy import copy
from typing import Any, TypeVar

T = TypeVar("T")

def copy_with_overrides(obj: T, **overrides: Any) -> T:
    instance = copy(obj)

    for key, value in overrides.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
        else:
            print(f"WARNING: Key {key} doesn't exist in {instance.__class__.__name__}")

    return instance
