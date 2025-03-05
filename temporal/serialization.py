from dataclasses import dataclass, field as datafield
from pathlib import Path
from typing import Any, Generic, Literal, Optional, Type, TypeVar, get_args

from temporal.utils.typing import get_optional_type, is_optional, safe_get_origin


T = TypeVar("T")


SerializationDataFlag = Literal["private", "runtime"]


@dataclass
class SerializationParams:
    data_dir: Optional[Path] = None
    flags: set[SerializationDataFlag] = datafield(default_factory = set)


def deserialize(type: Type[Any], obj: Any, params: SerializationParams) -> Any:
    if safe_get_origin(type) is list:
        return [
            deserialize(get_args(type)[0], value, params)
            for value in obj
        ]

    elif safe_get_origin(type) is dict:
        return {
            key: deserialize(get_args(type)[1], value, params)
            for key, value in obj.items()
        }

    elif safe_get_origin(type) is Any:
        return obj

    elif safe_get_origin(type) is Literal:
        return deserialize(str, obj, params)

    elif is_optional(type):
        return deserialize(get_optional_type(type) if obj is not None else NoneType, obj, params)

    elif serializer := _find_serializer(type):
        return serializer.read_json(obj, params)

    else:
        raise Exception(f"Couldn't find a serializer for '{type}'")


def serialize(type: Type[Any], obj: Any, params: SerializationParams) -> Any:
    if safe_get_origin(type) is list:
        return [
            serialize(get_args(type)[0], value, params)
            for value in obj
        ]

    elif safe_get_origin(type) is dict:
        return {
            key: serialize(get_args(type)[1], value, params)
            for key, value in obj.items()
        }

    elif safe_get_origin(type) is Any:
        return obj

    elif safe_get_origin(type) is Literal:
        return serialize(str, obj, params)

    elif is_optional(type):
        return serialize(get_optional_type(type) if obj is not None else NoneType, obj, params)

    elif serializer := _find_serializer(type):
        return serializer.write_json(obj, params)

    else:
        raise Exception(f"Couldn't find a serializer for '{type}'")


class Serializer(Generic[T]):
    def __init_subclass__(cls) -> None:
        serialized_type = cls.get_type()

        if serialized_type in _serializers:
            raise Exception(f"{serialized_type} is already registered")

        _serializers[serialized_type] = cls

    @classmethod
    def get_type(cls) -> Type[T]:
        return get_args(getattr(cls, "__orig_bases__")[0])[0]

    @classmethod
    def read_json(cls, obj: Any, params: SerializationParams) -> T:
        raise NotImplementedError

    @classmethod
    def write_json(cls, obj: T, params: SerializationParams) -> Any:
        raise NotImplementedError


def _find_serializer(type: Type[Any]) -> Optional[Type[Serializer[Any]]]:
    if (best_type := _serializers.get(type, None)) is not None:
        return best_type

    best_index = int(1e9)
    best_type = None

    mro = type.mro()

    for key, alias in _serializers.items():
        try:
            mro_index = mro.index(key)
        except ValueError:
            continue

        if mro_index < best_index:
            best_index = mro_index
            best_type = alias

    return best_type


_serializers: dict[Type[Any], Type[Serializer[Any]]] = {}


#===============================================================================


from types import NoneType

from temporal.utils.bytes import base64_to_bytes, bytes_to_base64
from temporal.utils.image import NumpyImage, PILImage, base64_to_image, image_to_base64, load_image, np_to_pil, pil_to_np, save_image
from temporal.utils.numpy import FloatArray, array_to_base64, base64_to_array, load_array, save_array


class _(Serializer[NoneType]):
    @classmethod
    def read_json(cls, obj, params):
        return obj

    @classmethod
    def write_json(cls, obj, params):
        return obj


class _(Serializer[bool]):
    @classmethod
    def read_json(cls, obj, params):
        return obj

    @classmethod
    def write_json(cls, obj, params):
        return obj


class _(Serializer[int]):
    @classmethod
    def read_json(cls, obj, params):
        return obj

    @classmethod
    def write_json(cls, obj, params):
        return obj


class _(Serializer[float]):
    @classmethod
    def read_json(cls, obj, params):
        return obj

    @classmethod
    def write_json(cls, obj, params):
        return obj


class _(Serializer[str]):
    @classmethod
    def read_json(cls, obj, params):
        return obj

    @classmethod
    def write_json(cls, obj, params):
        return obj


class _(Serializer[Path]):
    @classmethod
    def read_json(cls, obj, params):
        return Path(obj)

    @classmethod
    def write_json(cls, obj, params):
        return obj.as_posix()


class _(Serializer[bytes]):
    @classmethod
    def read_json(cls, obj, params):
        if params.data_dir is not None:
            return (params.data_dir / obj).read_bytes()
        else:
            return base64_to_bytes(obj)

    @classmethod
    def write_json(cls, obj, params):
        if params.data_dir is not None:
            path = params.data_dir / f"{id(obj)}.bin"
            path.write_bytes(obj)
            return path.name
        else:
            return bytes_to_base64(obj)


class _(Serializer[PILImage]):
    @classmethod
    def read_json(cls, obj, params):
        if params.data_dir is not None:
            return load_image(params.data_dir / obj)
        else:
            return np_to_pil(base64_to_image(obj))

    @classmethod
    def write_json(cls, obj, params):
        if params.data_dir is not None:
            path = params.data_dir / f"{id(obj)}.png"
            save_image(obj, path)
            return path.name
        else:
            return image_to_base64(pil_to_np(obj), "fast")


class _(Serializer[FloatArray]):
    @classmethod
    def read_json(cls, obj, params):
        if params.data_dir is not None:
            return load_array(params.data_dir / obj)
        else:
            return base64_to_array(obj)

    @classmethod
    def write_json(cls, obj, params):
        if params.data_dir is not None:
            path = params.data_dir / f"{id(obj)}.npz"
            save_array(obj, path)
            return path.name
        else:
            return array_to_base64(obj)


class _(Serializer[NumpyImage]):
    @classmethod
    def read_json(cls, obj, params):
        if params.data_dir is not None:
            return load_array(params.data_dir / obj)
        else:
            return base64_to_image(obj)

    @classmethod
    def write_json(cls, obj, params):
        if params.data_dir is not None:
            path = params.data_dir / f"{id(obj)}.npz"
            save_array(obj, path)
            return path.name
        else:
            return image_to_base64(obj)
