from copy import copy
from inspect import isclass, isfunction
from itertools import chain
from json import dumps, loads
from pathlib import Path
from typing import Any, Callable, Generic, Literal, Optional, Type, TypeVar, cast, get_args, get_type_hints
from typing_extensions import Self
from uuid import uuid4
from weakref import WeakValueDictionary

from temporal.serialization import SerializationDataFlag, SerializationParams, Serializer, deserialize, serialize
from temporal.utils import logging
from temporal.utils.fs import recreate_directory
from temporal.utils.typing import get_full_type_name, get_optional_type, is_optional, safe_get_origin


T = TypeVar("T")


class UndefinedValue:
    pass


class Static(Generic[T]):
    def __new__(cls, value: T | Type[UndefinedValue] = UndefinedValue) -> T:
        instance = object.__new__(cls)
        instance.__init__(value)
        return cast(T, instance)

    def __init__(self, value: T | Type[UndefinedValue] = UndefinedValue) -> None:
        self.key = ""
        self.type: Type[Any]
        self.value = value

    def __set_name__(self, owner: Any, name: str) -> None:
        self.key = name
        self.type = get_type_hints(owner, include_extras = True)[name]


class Meta(Static[T]):
    def __new__(cls, value: T | Type[UndefinedValue] = UndefinedValue) -> T:
        instance = object.__new__(cls)
        instance.__init__(value)
        return cast(T, instance)


class Field(Generic[T]):
    def __new__(
        cls,
        value: T | Callable[[], T] | Type[UndefinedValue] = UndefinedValue,
        *,
        flags: set[SerializationDataFlag] = set(),
    ) -> T:
        instance = object.__new__(cls)
        instance.__init__(value, flags = flags)
        return cast(T, instance)

    def __init__(
        self,
        value: T | Callable[[], T] | Type[UndefinedValue] = UndefinedValue,
        *,
        flags: set[SerializationDataFlag] = set(),
    ) -> None:
        self.key = ""
        self.type: Type[Any]
        self.value = value
        self.flags = flags

    def __set_name__(self, owner: Any, name: str) -> None:
        self.key = name
        self.type = get_type_hints(owner, include_extras = True)[name]

    @property
    def default(self) -> T:
        value = self.value

        if value is UndefinedValue:
            raise ValueError("No default value is provided")
        elif isfunction(value):
            return value()
        elif isclass(value):
            return cast(T, value())
        else:
            return cast(T, value)


Choices = list[T] | dict[T, str]
UIType = Literal["area", "box", "code", "menu", "radio", "seed", "slider"]


class Param(Field[T]):
    def __new__(
        cls,
        name: str = "Parameter",
        value: T | Callable[[], T] | Type[UndefinedValue] = UndefinedValue,
        *,
        minimum: Optional[int | float] = None,
        maximum: Optional[int | float] = None,
        step: Optional[int | float] = None,
        axes: Optional[list[str]] = None,
        channels: Optional[int] = None,
        choices: Optional[Choices[T]] | Callable[[], Choices[T]] = None,
        language: Optional[str] = None,
        dependencies: Optional[dict[str, Any]] = None,
        ui_type: Optional[UIType] = None,
    ) -> T:
        instance = object.__new__(cls)
        instance.__init__(
            name = name,
            value = value,
            minimum = minimum,
            maximum = maximum,
            step = step,
            axes = axes,
            channels = channels,
            choices = choices,
            language = language,
            dependencies = dependencies,
            ui_type = ui_type,
        )
        return cast(T, instance)

    def __init__(
        self,
        name: str = "Parameter",
        value: T | Callable[[], T] | Type[UndefinedValue] = UndefinedValue,
        *,
        minimum: Optional[int | float] = None,
        maximum: Optional[int | float] = None,
        step: Optional[int | float] = None,
        axes: Optional[list[str]] = None,
        channels: Optional[int] = None,
        choices: Optional[Choices[T]] | Callable[[], Choices[T]] = None,
        language: Optional[str] = None,
        dependencies: Optional[dict[str, Any]] = None,
        ui_type: Optional[UIType] = None,
    ) -> None:
        super().__init__(value = value)
        self.name = name
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.axes = axes
        self.channels = channels
        self.choices = choices
        self.language = language
        self.dependencies = dependencies
        self.ui_type = ui_type

    @property
    def schema(self) -> dict[str, Any]:
        choices = self.choices

        if choices is not None:
            choices = choices() if callable(choices) else choices

            if isinstance(choices, list):
                choices = {x: x for x in choices}

        type = self.type

        if is_optional(type):
            type = get_optional_type(type)

        if safe_get_origin(type) is Literal:
            type = str

        return {
            "type": get_full_type_name(type),
            "optional": is_optional(self.type),
            "subtype": get_full_type_name(get_args(type)[0]) if safe_get_origin(type) is list else None,
            "name": self.name,
            **({"minimum": self.minimum} if self.minimum is not None else {}),
            **({"maximum": self.maximum} if self.maximum is not None else {}),
            **({"step": self.step} if self.step is not None else {}),
            **({"axes": self.axes} if self.axes is not None else {}),
            **({"channels": self.channels} if self.channels is not None else {}),
            **({"choices": choices} if choices is not None else {}),
            **({"language": self.language} if self.language is not None else {}),
            **({"dependencies": {k: serialize(v.__class__, v, SerializationParams()) for k, v in self.dependencies.items()}} if self.dependencies else {}),
            **({"ui_type": self.ui_type} if self.ui_type is not None else {}),
            **({"default": serialize(self.type, self.default, SerializationParams())} if self.default is not None else {}),
        }


class Object:
    __type_name__: str
    __statics__: dict[str, Static[Any]] = {}
    __metas__: dict[str, Meta[Any]] = {}
    __fields__: dict[str, Field[Any]] = {}
    __params__: dict[str, Param[Any]] = {}
    __subtypes__: list[Type[Self]] = []

    def __init_subclass__(
        cls,
        *,
        abstract: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)

        class _(Serializer[cls]):
            @classmethod
            def read_json(cls, obj, params):
                return cls.get_type().from_json(obj, params)

            @classmethod
            def write_json(cls, obj, params):
                return obj.to_json(params)

        cls.__type_name__ = get_full_type_name(cls)
        cls.__statics__ = {
            key: copy(field)
            for base_cls in list(reversed(cls.__mro__[1:]))
            if issubclass(base_cls, Object)
            for key, field in base_cls.__statics__.items()
        } | {
            key: field
            for key, field in cls.__dict__.items()
            if isinstance(field, Static)
        }
        cls.__metas__ = {
            key: field
            for key, field in cls.__statics__.items()
            if isinstance(field, Meta)
        }
        cls.__fields__ = {
            key: field
            for base_cls in list(reversed(cls.__mro__[1:]))
            if issubclass(base_cls, Object)
            for key, field in base_cls.__fields__.items()
        } | {
            key: field
            for key, field in cls.__dict__.items()
            if isinstance(field, Field)
        }
        cls.__params__ = {
            key: field
            for key, field in cls.__fields__.items()
            if isinstance(field, Param)
        }
        cls.__subtypes__ = []

        if not abstract:
            for base_cls in list(cls.__mro__):
                if issubclass(base_cls, Object):
                    base_cls.__subtypes__.append(cls)

        for key, field in cls.__statics__.items():
            field.value = value if not isinstance(value := getattr(cls, key), Static) else field.value
            setattr(cls, key, field.value)

        object_types[cls.__type_name__] = cls

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        initialized_keys = set()

        for key, value in chain(zip(self.__fields__.keys(), args), kwargs.items()):
            setattr(self, key, value)
            initialized_keys.add(key)

        for key, field in self.__fields__.items():
            if key not in initialized_keys:
                setattr(self, key, field.default)

        if not getattr(self, "__id__", None):
            self.__id__ = str(uuid4())

        object_registry[self.__id__] = self

    def __repr__(self) -> str:
        args = ", ".join(f"{key} = {repr(getattr(self, key))}" for key in self.__fields__.keys())
        return f"{self.__class__.__name__}({args})"

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return {
            "type": cls.__type_name__,
            **{k: v.value for k, v in cls.__metas__.items()},
            "parameters": {
                key: param.schema
                for key, param in cls.__params__.items()
            },
        }

    @classmethod
    def from_json(cls, data: dict[str, Any], params: SerializationParams = SerializationParams()) -> Self:
        if type_name := data.pop("__type__", None):
            if not issubclass(actual_cls := object_types[type_name], cls):
                raise TypeError
        else:
            actual_cls = cls

        return actual_cls(__id__ = data.pop("__id__", None), **{
            key: deserialize(field.type, data[key], params)
            for key, field in actual_cls.__fields__.items()
            if key in data and field.flags.issubset(params.flags)
        })

    def to_json(self, params: SerializationParams = SerializationParams()) -> dict[str, Any]:
        return {"__type__": self.__type_name__, "__id__": self.__id__} | {
            key: serialize(field.type, getattr(self, key), params)
            for key, field in self.__fields__.items()
            if field.flags.issubset(params.flags)
        }

    @classmethod
    def load(cls, dir: Path) -> Self:
        if not dir.is_dir() or not (json_path := dir / "data.json").is_file():
            logging.warning(f"Cannot load {cls.__name__} from {dir.as_posix()}")
            return cls()

        return cls.from_json(loads(json_path.read_text()), SerializationParams(data_dir = dir, flags = {"private"}))

    def save(self, dir: Path) -> None:
        if dir == Path("."):
            raise Exception

        recreate_directory(dir)

        (dir / "data.json").write_text(dumps(self.to_json(SerializationParams(data_dir = dir, flags = {"private"})), indent = 4))


object_types: dict[str, Type[Object]] = {}
object_registry: WeakValueDictionary[str, Object] = WeakValueDictionary()
