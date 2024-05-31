import xml.etree.ElementTree as ET
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from temporal.serialization import Archive, Serializer, find_alias_for_type
from temporal.utils import logging
from temporal.utils.fs import recreate_directory


T = TypeVar("T")


class SerializableField:
    def __init__(self, default: Optional[T] = None, factory: Optional[Callable[[], T]] = None, saved: bool = True) -> None:
        self.key = ""
        self.default = default
        self.factory = factory
        self.saved = saved

    def __set_name__(self, owner: Any, name: str) -> None:
        self.key = name

    def set_default_value(self, instance: Any) -> None:
        setattr(instance, self.key, self.factory() if self.factory is not None else self.default)


def field(default: Optional[T] = None, *, factory: Optional[Callable[[], T]] = None, saved: bool = True) -> Any:
    return SerializableField(default, factory, saved)


class Serializable:
    __fields__: dict[str, SerializableField]

    def __init_subclass__(cls, init: bool = True) -> None:
        class _(Serializer[cls]):
            @classmethod
            def read(cls, obj, ar):
                return obj.read(ar) or obj

            @classmethod
            def write(cls, obj, ar):
                obj.write(ar)

        fields = {
            key: field
            for base_cls in reversed(list(cls.__mro__) + [cls])
            if issubclass(base_cls, Serializable)
            for key, field in base_cls.__dict__.items()
            if isinstance(field, SerializableField)
        }

        setattr(cls, "__fields__", fields)

        if not init:
            return

        real_init_func = getattr(cls, "__init__", None)

        def new_init_func(self, *args: Any, **kwargs: Any) -> None:
            initialized_keys = set()

            for key, value in chain(zip(fields.keys(), args), kwargs.items()):
                setattr(self, key, value)
                initialized_keys.add(key)

            for key, field in fields.items():
                if key not in initialized_keys:
                    field.set_default_value(self)

            if real_init_func is not None:
                real_init_func(self, *args, **kwargs)

        setattr(cls, "__init__", new_init_func)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

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
        ar = Archive(data_dir = dir)
        ar.parse_xml(ET.ElementTree(file = dir / "data.xml").getroot())
        self.read(ar)

    def save(self, dir: Path) -> None:
        if dir == Path("."):
            raise Exception

        recreate_directory(dir)

        ar = Archive(type_name = find_alias_for_type(type(self)) or "", data_dir = dir)
        self.write(ar)

        tree = ET.ElementTree(ar.print_xml())
        ET.indent(tree)
        tree.write(dir / "data.xml")
