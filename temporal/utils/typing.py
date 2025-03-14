from dataclasses import dataclass
from inspect import getmodule, stack
from types import NoneType
from typing import Annotated, Any, Optional, Type, Union, get_args, get_origin

from temporal.utils.collection import find_by_predicate


@dataclass(unsafe_hash = True)
class Alias:
    name: str
    module_name: Optional[str] = None

    def __post_init__(self) -> None:
        if module := getmodule(stack()[2].frame):
            self.module_name = module.__name__


def get_alias_module_name(type: Type[Any]) -> Optional[str]:
    if alias := find_by_predicate(reversed(get_args(type)), lambda x: isinstance(x, Alias)):
        return alias.module_name


def get_alias_name(type: Type[Any]) -> str:
    if alias := find_by_predicate(reversed(get_args(type)), lambda x: isinstance(x, Alias)):
        return alias.name
    else:
        raise TypeError


def get_full_type_name(type: Type[Any]) -> str:
    if is_alias(type):
        if (module_name := get_alias_module_name(type)):
            return f"{module_name}.{get_alias_name(type)}"
        else:
            return get_alias_name(type)
    elif type.__module__ != "builtins":
        return f"{type.__module__}.{getattr(type, '__qualname__', type.__name__)}"
    else:
        return type.__name__


def get_optional_type(type: Type[Any]) -> Type[Any]:
    if result := find_by_predicate(get_args(type), lambda x: x is not NoneType):
        return result
    else:
        raise TypeError


def is_alias(type: Type[Any]) -> bool:
    return get_origin(type) is Annotated and find_by_predicate(get_args(type), lambda x: isinstance(x, Alias)) is not None


def is_optional(type: Type[Any]) -> bool:
    return get_origin(type) is Union and len(args := get_args(type)) == 2 and NoneType in args


def safe_get_origin(type: Type[Any]) -> Any:
    return origin if (origin := get_origin(type)) is not None else type
