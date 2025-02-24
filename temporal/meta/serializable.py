import xml.etree.ElementTree as ET
from itertools import chain
from pathlib import Path
from types import NoneType
from typing import Any, Callable, Literal, Optional, Type, TypeVar, cast, get_args, get_type_hints

from temporal.serialization import Archive, Serializer, find_alias_for_type, find_serializer
from temporal.utils import logging
from temporal.utils.fs import recreate_directory
from temporal.utils.typing import get_optional_type, is_optional, safe_get_origin


T = TypeVar("T")
U = TypeVar("U", bound = "Serializable")


SerializableFieldFlag = Literal["private", "runtime"]


class SerializableField:
    def __new__(cls, value: Optional[T] = None, *, factory: Optional[Callable[[], T]] = None, variant: str = "", flags: set[SerializableFieldFlag] = set()) -> T:
        instance = object.__new__(cls)
        instance.__init__(value, factory = factory, flags = flags, variant = variant)
        return cast(T, instance)

    def __init__(self, value: Optional[T] = None, *, factory: Optional[Callable[[], T]] = None, variant: str = "", flags: set[SerializableFieldFlag] = set()) -> None:
        self.key = ""
        self.type: Type[Any]
        self.value = value
        self.factory = factory
        self.variant = variant
        self.flags = flags

    def __set_name__(self, owner: Any, name: str) -> None:
        self.key = name
        self.type = get_type_hints(owner, include_extras = True)[name]

    @property
    def default(self) -> Any:
        return self.factory() if self.factory is not None else self.value


class Serializable:
    __type_name__: str
    __fields__: dict[str, SerializableField]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        cls.__type_name__ = f"{cls.__module__}.{getattr(cls, '__qualname__', cls.__name__)}"
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
        args = ", ".join(f"{key} = {getattr(self, key)}" for key in self.__fields__.keys())
        return f"{self.__class__.__name__}({args})"

    def read(self, ar: Archive) -> None:
        for child in ar:
            try:
                current_value = getattr(self, child.key)
            except AttributeError:
                logging.warning(f"{child.key} is not found in class {self.__class__.__name__}")
                continue

            setattr(self, child.key, child.read(current_value))

    def write(self, ar: Archive) -> None:
        for key, field in self.__fields__.items():
            if "runtime" not in field.flags:
                ar[key].write(getattr(self, key))

    def load(self, dir: Path) -> None:
        # FIXME
        # if not dir.is_dir() or not (xml_path := dir / "data.xml").is_file():
        #     logging.warning(f"Cannot load {self.__class__.__name__} from {dir.as_posix()}")
        #     return

        # self.from_xml(ET.ElementTree(file = xml_path).getroot(), data_dir = dir)

        from json import loads

        if not dir.is_dir() or not (json_path := dir / "data.json").is_file():
            logging.warning(f"Cannot load {self.__class__.__name__} from {dir.as_posix()}")
            return

        loaded_obj = self.from_json(loads(json_path.read_text()))

        for key in loaded_obj.__fields__.keys():
            self.__dict__[key] = loaded_obj.__dict__[key]

    def save(self, dir: Path) -> None:
        if dir == Path("."):
            raise Exception

        recreate_directory(dir)

        # FIXME
        # tree = self.to_xml(data_dir = dir)
        # tree.write(dir / "data.xml")

        from json import dumps

        (dir / "data.json").write_text(dumps(self.to_json(include_flags = {"private"}), indent = 4))

    @classmethod
    def from_json(cls: Type[U], data: dict[str, Any]) -> U:
        def read_value(type: Type[Any], obj: Any, variant: str = "") -> Any:
            if safe_get_origin(type) is list:
                return [
                    read_value(get_args(type)[0], value)
                    for value in obj
                ]

            elif safe_get_origin(type) is dict:
                return {
                    key: read_value(get_args(type)[1], value)
                    for key, value in obj.items()
                }

            elif safe_get_origin(type) is Literal:
                return read_value(str, obj, variant)

            elif is_optional(type):
                return read_value(get_optional_type(type) if obj is not None else NoneType, obj, variant if obj is not None else "")

            elif issubclass(type, Serializable):
                return type.from_json(obj)

            elif serializer := find_serializer(type, variant):
                return serializer.read_json(obj)

            else:
                raise Exception(f"Couldn't find a serializer for '{type}'")

        if "__type__" in data:
            cls = cast(Type[U], _serializables[data["__type__"]])

        return cls(**{
            key: read_value(field.type, data[key], field.variant)
            for key, field in cls.__fields__.items()
            if key in data
        })

    def to_json(self, include_flags: set[SerializableFieldFlag] = set()) -> dict[str, Any]:
        def write_value(type: Type[Any], obj: Any, variant: str = "") -> Any:
            if safe_get_origin(type) is list:
                return [
                    write_value(get_args(type)[0], value)
                    for value in obj
                ]

            elif safe_get_origin(type) is dict:
                return {
                    key: write_value(get_args(type)[1], value)
                    for key, value in obj.items()
                }

            elif safe_get_origin(type) is Literal:
                return write_value(str, obj, variant)

            elif is_optional(type):
                return write_value(get_optional_type(type) if obj is not None else NoneType, obj, variant if obj is not None else "")

            elif issubclass(type, Serializable):
                return obj.to_json(include_flags)

            elif serializer := find_serializer(type, variant):
                return serializer.write_json(obj)

            else:
                raise Exception(f"Couldn't find a serializer for '{type}'")

        return {"__type__": self.__type_name__} | {
            key: write_value(field.type, self.__dict__[key], field.variant)
            for key, field in self.__fields__.items()
            if field.flags.issubset(include_flags)
        }

    def from_xml(self: U, elem: ET.Element, data_dir: Optional[Path] = None) -> U:
        ar = Archive(data_dir = data_dir)
        ar.parse_xml(elem)
        self.read(ar)

        return self

    def to_xml(self, data_dir: Optional[Path] = None) -> ET.ElementTree:
        ar = Archive(type_name = find_alias_for_type(type(self)) or "", data_dir = data_dir)
        self.write(ar)

        tree = ET.ElementTree(ar.print_xml())
        ET.indent(tree)

        return tree


_serializables: dict[str, Type[Serializable]] = {}
