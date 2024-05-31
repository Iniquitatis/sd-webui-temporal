import xml.etree.ElementTree as ET
from dataclasses import dataclass, field as datafield
from pathlib import Path
from typing import Any, Generic, Iterator, Optional, Type, TypeVar, get_args, get_origin

from temporal.utils.object import get_property_by_path, set_property_by_path


T = TypeVar("T")


@dataclass(slots = True)
class Archive:
    key: str = "_"
    type_name: str = ""
    data: str = ""
    data_dir: Optional[Path] = None
    _children: list["Archive"] = datafield(default_factory = list, init = False)

    def __getitem__(self, key: int | str) -> "Archive":
        if (child := self._find_child(key)) is None:
            child = self.make_child()

            if isinstance(key, str):
                child.key = key

        return child

    def __iter__(self) -> Iterator["Archive"]:
        yield from self._children

    def make_child(self) -> "Archive":
        child = Archive(data_dir = self.data_dir)
        self._children.append(child)
        return child

    def create(self) -> Any:
        if self.type_name in _serializers:
            return _serializers[self.type_name].create(self)
        else:
            raise NotImplementedError

    def read(self, obj: T) -> T:
        if self.type_name != find_alias_for_type(type(obj)):
            return _serializers[self.type_name].create(self)
        elif self.type_name in _serializers:
            return _serializers[self.type_name].read(obj, self)
        else:
            raise NotImplementedError

    def write(self, obj: Any) -> "Archive":
        if (type_name := find_alias_for_type(type(obj))) is not None:
            self.type_name = type_name
            _serializers[type_name].write(obj, self)
        else:
            raise NotImplementedError

        return self

    def parse_xml(self, elem: ET.Element) -> None:
        self.key = elem.get("key", "_")

        if (type := elem.get("type", None)) is not None:
            self.type_name = type

        self.data = elem.text or ""

        for child_elem in elem:
            self.make_child().parse_xml(child_elem)

    def print_xml(self) -> ET.Element:
        attrs = {}

        if self.key != "_":
            attrs["key"] = self.key

        attrs["type"] = self.type_name

        elem = ET.Element("object", attrs)
        elem.text = self.data

        for child in self._children:
            elem.append(child.print_xml())

        return elem

    def _find_child(self, key: int | str) -> Optional["Archive"]:
        if isinstance(key, int):
            return self._children[key]
        else:
            for child in self._children:
                if child.key == key:
                    return child


_type_aliases: dict[Type[Any], str] = {}
_serializers: dict[str, Type["Serializer[Any]"]] = {}


class Serializer(Generic[T]):
    def __init_subclass__(cls, alias: Optional[str] = None) -> None:
        _type_aliases[cls.get_type()] = actual_alias = alias if alias is not None else getattr(cls.get_type(), "__name__", "_")
        _serializers[actual_alias] = cls

    @classmethod
    def get_type(cls) -> Type[T]:
        type = get_args(getattr(cls, "__orig_bases__")[0])[0]

        if (origin := get_origin(type)) is not None:
            return origin
        else:
            return type

    @classmethod
    def create(cls, ar: Archive) -> T:
        return cls.read(cls.get_type()(), ar)

    @classmethod
    def read(cls, obj: T, ar: Archive) -> T:
        raise NotImplementedError

    @classmethod
    def write(cls, obj: T, ar: Archive) -> None:
        raise NotImplementedError


class BasicObjectSerializer(Serializer[T]):
    keys: list[Any] = []

    def __init_subclass__(cls, alias: Optional[str] = None, create: bool = True) -> None:
        super().__init_subclass__(alias)

        if not create:
            setattr(cls, "create", lambda ar: None)

    @classmethod
    def read(cls, obj: T, ar: Archive) -> T:
        for key in cls.keys:
            cls._create_value(obj, ar, key)

        return obj

    @classmethod
    def write(cls, obj: T, ar: Archive) -> None:
        for key in cls.keys:
            cls._write_value(obj, ar, key)

    @staticmethod
    def _create_value(obj: Any, ar: Archive, key: str) -> None:
        set_property_by_path(obj, key, ar[key].create())

    @staticmethod
    def _write_value(obj: Any, ar: Archive, key: str) -> None:
        ar[key].write(get_property_by_path(obj, key))


def find_alias_for_type(type: Type[Any]) -> Optional[str]:
    best_index = int(1e9)
    best_alias = None

    mro = type.mro()

    for key, alias in _type_aliases.items():
        try:
            mro_index = mro.index(key)
        except ValueError:
            continue

        if mro_index < best_index:
            best_index = mro_index
            best_alias = alias

    return best_alias


#===============================================================================


from types import NoneType, SimpleNamespace

import numpy as np
from PIL import Image
from numpy.typing import NDArray

from temporal.utils.image import load_image, save_image
from temporal.utils.numpy import load_array, save_array


class _(Serializer[NoneType]):
    @classmethod
    def read(cls, obj, ar):
        return None

    @classmethod
    def write(cls, obj, ar):
        ar.data = ""


class _(Serializer[bool]):
    @classmethod
    def read(cls, obj, ar):
        return ar.data.lower() == "true"

    @classmethod
    def write(cls, obj, ar):
        ar.data = str(obj)


class _(Serializer[int]):
    @classmethod
    def read(cls, obj, ar):
        return int(ar.data)

    @classmethod
    def write(cls, obj, ar):
        ar.data = str(obj)


class _(Serializer[float]):
    @classmethod
    def read(cls, obj, ar):
        return float(ar.data)

    @classmethod
    def write(cls, obj, ar):
        ar.data = str(obj)


class _(Serializer[str]):
    @classmethod
    def read(cls, obj, ar):
        return ar.data

    @classmethod
    def write(cls, obj, ar):
        ar.data = obj


class _(Serializer[tuple[Any, ...]]):
    @classmethod
    def read(cls, obj, ar):
        return tuple(x.create() for x in ar)

    @classmethod
    def write(cls, obj, ar):
        for value in obj:
            ar.make_child().write(value)


class _(Serializer[list[Any]]):
    @classmethod
    def read(cls, obj, ar):
        obj[:] = [x.create() for x in ar]
        return obj

    @classmethod
    def write(cls, obj, ar):
        for value in obj:
            ar.make_child().write(value)


class _(Serializer[set[Any]]):
    @classmethod
    def read(cls, obj, ar):
        obj.clear()
        obj |= {x.create() for x in ar}
        return obj

    @classmethod
    def write(cls, obj, ar):
        for value in obj:
            ar.make_child().write(value)


class _(Serializer[dict[Any, Any]]):
    @classmethod
    def read(cls, obj, ar):
        obj.clear()
        obj |= {x.key: x.create() for x in ar}
        return obj

    @classmethod
    def write(cls, obj, ar):
        for key, value in obj.items():
            ar[key].write(value)


class _(Serializer[Path]):
    @classmethod
    def read(cls, obj, ar):
        return Path(ar.data)

    @classmethod
    def write(cls, obj, ar):
        ar.data = obj.as_posix()


class _(Serializer[SimpleNamespace]):
    @classmethod
    def read(cls, obj, ar):
        for child_ar in ar:
            setattr(obj, child_ar.key, child_ar.create())

        return obj

    @classmethod
    def write(cls, obj, ar):
        for key, value in vars(obj).items():
            ar[key].write(value)


class _(Serializer[Image.Image]):
    @classmethod
    def read(cls, obj, ar):
        if ar.data_dir is not None:
            return load_image(ar.data_dir / ar.data)
        else:
            raise NotADirectoryError

    @classmethod
    def write(cls, obj, ar):
        if ar.data_dir is not None:
            path = ar.data_dir / f"{id(obj)}.png"
            save_image(obj, path)
            ar.data = path.name
        else:
            raise NotADirectoryError


class _(Serializer[NDArray[np.float_]]):
    @classmethod
    def read(cls, obj, ar):
        if ar.data_dir is not None:
            return load_array(ar.data_dir / ar.data)
        else:
            raise NotADirectoryError

    @classmethod
    def write(cls, obj, ar):
        if ar.data_dir is not None:
            path = ar.data_dir / f"{id(obj)}.npz"
            save_array(obj, path)
            ar.data = path.name
        else:
            raise NotADirectoryError
