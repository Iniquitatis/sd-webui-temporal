from types import NoneType
from typing import Any, Type, Union, get_args, get_origin


def get_full_type_name(type: Type[Any]) -> str:
    if type.__module__ != "builtins":
        return f"{type.__module__}.{getattr(type, '__qualname__', type.__name__)}"
    else:
        return type.__name__


def get_optional_type(type: Type[Any]) -> Type[Any]:
    for arg in get_args(type):
        if arg is not NoneType:
            return arg
    else:
        raise TypeError


def is_optional(type: Type[Any]) -> bool:
    return get_origin(type) is Union and NoneType in get_args(type)


def safe_get_origin(type: Type[Any]) -> Any:
    origin = get_origin(type)
    return origin if origin is not None else type
