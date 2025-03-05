from inspect import isclass, isfunction
from itertools import chain
from json import dumps, loads
from pathlib import Path
from typing import Any, Callable, Generic, Type, TypeVar, cast, get_type_hints

from temporal.serialization import SerializationDataFlag, SerializationParams, Serializer, deserialize, serialize
from temporal.utils import logging
from temporal.utils.fs import recreate_directory
from temporal.utils.typing import get_full_type_name


T = TypeVar("T")
U = TypeVar("U", bound = "Serializable")


class UndefinedValue:
    pass


class SerializableField(Generic[T]):
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


class Serializable:
    __type_name__: str
    __fields__: dict[str, SerializableField[Any]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        class _(Serializer[cls]):
            @classmethod
            def read_json(cls, obj, params):
                return cls.get_type().from_json(obj, params)

            @classmethod
            def write_json(cls, obj, params):
                return obj.to_json(params)

        cls.__type_name__ = get_full_type_name(cls)
        cls.__fields__ = {
            key: field
            for base_cls in list(reversed(cls.__mro__)) + [cls]
            if issubclass(base_cls, Serializable)
            for key, field in base_cls.__dict__.items()
            if isinstance(field, SerializableField)
        }

        _serializables[cls.__type_name__] = cls

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        initialized_keys = set()

        for key, value in chain(zip(self.__fields__.keys(), args), kwargs.items()):
            setattr(self, key, value)
            initialized_keys.add(key)

        for key, field in self.__fields__.items():
            if key not in initialized_keys:
                setattr(self, key, field.default)

    def __repr__(self) -> str:
        args = ", ".join(f"{key} = {repr(getattr(self, key))}" for key in self.__fields__.keys())
        return f"{self.__class__.__name__}({args})"

    @classmethod
    def from_json(cls: Type[U], data: dict[str, Any], params: SerializationParams = SerializationParams()) -> U:
        if "__type__" in data:
            cls = cast(Type[U], _serializables[data["__type__"]])

        return cls(**{
            key: deserialize(field.type, data[key], params)
            for key, field in cls.__fields__.items()
            if key in data and field.flags.issubset(params.flags)
        })

    def to_json(self, params: SerializationParams = SerializationParams()) -> dict[str, Any]:
        return {"__type__": self.__type_name__} | {
            key: serialize(field.type, getattr(self, key), params)
            for key, field in self.__fields__.items()
            if field.flags.issubset(params.flags)
        }

    @classmethod
    def load(cls: Type[U], dir: Path) -> U:
        if not dir.is_dir() or not (json_path := dir / "data.json").is_file():
            logging.warning(f"Cannot load {cls.__name__} from {dir.as_posix()}")
            return cls()

        return cls.from_json(loads(json_path.read_text()), SerializationParams(data_dir = dir, flags = {"private"}))

    def save(self, dir: Path) -> None:
        if dir == Path("."):
            raise Exception

        recreate_directory(dir)

        (dir / "data.json").write_text(dumps(self.to_json(SerializationParams(data_dir = dir, flags = {"private"})), indent = 4))


_serializables: dict[str, Type[Serializable]] = {}
