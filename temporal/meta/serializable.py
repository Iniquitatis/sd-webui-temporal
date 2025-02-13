import xml.etree.ElementTree as ET
from itertools import chain
from pathlib import Path
from types import NoneType
from typing import Any, Callable, Literal, Optional, Type, TypeVar, cast, get_args, get_type_hints

from temporal.serialization import Archive, Serializer, Variant, find_alias_for_type, find_serializer
from temporal.utils import logging
from temporal.utils.fs import recreate_directory
from temporal.utils.typing import get_annotated_arg, get_annotated_type, get_optional_type, is_annotated, is_optional, safe_get_origin


T = TypeVar("T")
U = TypeVar("U", bound = "Serializable")


class SerializableField:
    def __new__(cls, value: Optional[T] = None, *, factory: Optional[Callable[[], T]] = None, saved: bool = True) -> T:
        instance = object.__new__(cls)
        instance.__init__(value, factory = factory, saved = saved)
        return cast(T, instance)

    def __init__(self, value: Optional[T] = None, *, factory: Optional[Callable[[], T]] = None, saved: bool = True) -> None:
        self.key = ""
        self.type: Type[Any]
        self.value = value
        self.factory = factory
        self.saved = saved

    def __set_name__(self, owner: Any, name: str) -> None:
        self.key = name
        self.type = get_type_hints(owner, include_extras = True)[name]

    @property
    def default(self) -> Any:
        return self.factory() if self.factory is not None else self.value


class Serializable:
    __fields__: dict[str, SerializableField]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        class _(Serializer[cls]):
            @classmethod
            def read(cls, obj, ar):
                return obj.read(ar) or obj

            @classmethod
            def write(cls, obj, ar):
                obj.write(ar)

            @classmethod
            def read_json(cls, obj):
                return cls.get_type().from_json(obj)

            @classmethod
            def write_json(cls, obj):
                return obj.to_json()

        cls.__fields__ = {
            key: field
            for base_cls in list(reversed(cls.__mro__)) + [cls]
            if issubclass(base_cls, Serializable)
            for key, field in base_cls.__dict__.items()
            if isinstance(field, SerializableField)
        }

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
            if field.saved:
                ar[key].write(getattr(self, key))

    def load(self, dir: Path) -> None:
        if not dir.is_dir() or not (xml_path := dir / "data.xml").is_file():
            logging.warning(f"Cannot load {self.__class__.__name__} from {dir.as_posix()}")
            return

        self.from_xml(ET.ElementTree(file = xml_path).getroot(), data_dir = dir)

    def save(self, dir: Path) -> None:
        if dir == Path("."):
            raise Exception

        recreate_directory(dir)

        tree = self.to_xml(data_dir = dir)
        tree.write(dir / "data.xml")

    @classmethod
    def from_json(cls: Type[U], data: dict[str, Any]) -> U:
        def read_value(type: Type[Any], obj: Any, variant: Variant = Variant()) -> Any:
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
                return read_value(str, obj)

            elif is_optional(type):
                return read_value(get_optional_type(type) if obj is not None else NoneType, obj)

            elif is_annotated(type):
                return read_value(get_annotated_type(type), obj, get_annotated_arg(type, Variant))

            elif serializer := find_serializer(type, variant):
                print(serializer.get_type())
                return serializer.read_json(obj)

            else:
                raise Exception(f"Couldn't find a serializer for '{type}'")

        return cls(**{
            key: read_value(field.type, data[key])
            for key, field in cls.__fields__.items()
            if key in data
        })

    def to_json(self) -> dict[str, Any]:
        def write_value(type: Type[Any], obj: Any, variant: Variant = Variant()) -> Any:
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
                return write_value(str, obj)

            elif is_optional(type):
                return write_value(get_optional_type(type) if obj is not None else NoneType, obj)

            elif is_annotated(type):
                return write_value(get_annotated_type(type), obj, get_annotated_arg(type, Variant))

            elif serializer := find_serializer(type, variant):
                return serializer.write_json(obj)

            else:
                raise Exception(f"Couldn't find a serializer for '{type}'")

        return {
            key: write_value(field.type, self.__dict__[key])
            for key, field in self.__fields__.items()
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
