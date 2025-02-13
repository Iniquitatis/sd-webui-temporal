from types import NoneType
from typing import Annotated, Any, Type, TypeVar, Union, get_args, get_origin


T = TypeVar("T")


def get_annotated_arg(type: Type[Any], arg_type: Type[T]) -> T:
    for arg in get_args(type)[1:]:
        if isinstance(arg, arg_type):
            return arg
    else:
        raise ValueError


def get_annotated_args(type: Type[Any]) -> tuple[Any, ...]:
    return get_args(type)[1:]


def get_annotated_type(type: Type[Any]) -> Type[Any]:
    return get_args(type)[0]


def get_optional_type(type: Type[Any]) -> Type[Any]:
    for arg in get_args(type):
        if arg is not NoneType:
            return arg
    else:
        raise TypeError


def is_annotated(type: Type[Any]) -> bool:
    return get_origin(type) is Annotated


def is_optional(type: Type[Any]) -> bool:
    return get_origin(type) is Union and NoneType in get_args(type)


def safe_get_origin(type: Type[Any]) -> Any:
    origin = get_origin(type)
    return origin if origin is not None else type
