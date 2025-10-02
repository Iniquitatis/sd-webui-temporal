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


# NOTE: `Any` because type checkers aren't good with recursive types
JSONValue = None | bool | int | float | str | list[Any] | dict[str, Any]


def deserialize(type: Type[Any], obj: JSONValue, params: SerializationParams) -> Any:
    if safe_get_origin(type) is list and isinstance(obj, list):
        return [
            deserialize(get_args(type)[0], value, params)
            for value in obj
        ]

    elif safe_get_origin(type) is dict and isinstance(obj, dict):
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


def serialize(type: Type[Any], obj: Any, params: SerializationParams) -> JSONValue:
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


def validate(type: Type[Any], obj: Any, criteria: dict[str, Any]) -> Any:
    if safe_get_origin(type) is list:
        return [
            validate(get_args(type)[0], value, criteria)
            for value in obj
        ]

    elif safe_get_origin(type) is dict:
        return {
            key: validate(get_args(type)[1], value, criteria)
            for key, value in obj.items()
        }

    elif safe_get_origin(type) is Any:
        return obj

    elif safe_get_origin(type) is Literal:
        return validate(str, obj, criteria)

    elif is_optional(type):
        return validate(get_optional_type(type) if obj is not None else NoneType, obj, criteria)

    elif serializer := _find_serializer(type):
        return serializer.validate(obj, criteria)

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
    def read_json(cls, obj: JSONValue, params: SerializationParams) -> T:
        raise NotImplementedError

    @classmethod
    def write_json(cls, obj: T, params: SerializationParams) -> JSONValue:
        raise NotImplementedError

    @classmethod
    def validate(cls, obj: T, criteria: dict[str, Any]) -> T:
        raise NotImplementedError


def _find_serializer(type: Type[Any]) -> Optional[Type[Serializer[Any]]]:
    if (serializer := _serializers.get(type, None)) is not None:
        return serializer

    best_index = int(1e9)
    best_serializer = None

    mro = type.mro()

    for serialized_type, serializer in _serializers.items():
        try:
            mro_index = mro.index(serialized_type)
        except ValueError:
            continue

        if mro_index < best_index:
            best_index = mro_index
            best_serializer = serializer

    return best_serializer


_serializers: dict[Type[Any], Type[Serializer[Any]]] = {}


#===============================================================================


from types import NoneType

from temporal.utils.bytes import base64_to_bytes, bytes_to_base64
from temporal.utils.image import NumpyImage, PILImage, base64_to_image, ensure_image_dims, image_to_base64, load_image, np_to_pil, pil_to_np, save_image
from temporal.utils.numpy import FloatArray, array_to_base64, base64_to_array, load_array, save_array


class _(Serializer[NoneType]):
    @classmethod
    def read_json(cls, obj, params):
        if obj is not None:
            raise ValueError

        return obj

    @classmethod
    def write_json(cls, obj, params):
        return obj

    @classmethod
    def validate(cls, obj, criteria):
        return obj


class _(Serializer[bool]):
    @classmethod
    def read_json(cls, obj, params):
        if not isinstance(obj, bool):
            raise ValueError

        return obj

    @classmethod
    def write_json(cls, obj, params):
        return obj

    @classmethod
    def validate(cls, obj, criteria):
        return obj


class _(Serializer[int]):
    @classmethod
    def read_json(cls, obj, params):
        if not isinstance(obj, (bool, int)):
            raise ValueError

        return int(obj)

    @classmethod
    def write_json(cls, obj, params):
        return obj

    @classmethod
    def validate(cls, obj, criteria):
        if (minimum := criteria.get("minimum")) is not None and obj < minimum:
            obj = minimum

        if (maximum := criteria.get("maximum")) is not None and obj > maximum:
            obj = maximum

        return obj


class _(Serializer[float]):
    @classmethod
    def read_json(cls, obj, params):
        if not isinstance(obj, (bool, int, float)):
            raise ValueError

        return float(obj)

    @classmethod
    def write_json(cls, obj, params):
        return obj

    @classmethod
    def validate(cls, obj, criteria):
        if (minimum := criteria.get("minimum")) is not None and obj < minimum:
            obj = minimum

        if (maximum := criteria.get("maximum")) is not None and obj > maximum:
            obj = maximum

        return obj


class _(Serializer[str]):
    @classmethod
    def read_json(cls, obj, params):
        if not isinstance(obj, str):
            raise ValueError

        return obj

    @classmethod
    def write_json(cls, obj, params):
        return obj

    @classmethod
    def validate(cls, obj, criteria):
        return obj


class _(Serializer[Path]):
    @classmethod
    def read_json(cls, obj, params):
        if not isinstance(obj, str):
            raise ValueError

        return Path(obj)

    @classmethod
    def write_json(cls, obj, params):
        return obj.as_posix()

    @classmethod
    def validate(cls, obj, criteria):
        return obj


class _(Serializer[bytes]):
    @classmethod
    def read_json(cls, obj, params):
        if not isinstance(obj, str):
            raise ValueError

        if params.data_dir is not None:
            return (params.data_dir / obj).read_bytes()
        else:
            return base64_to_bytes(obj, True)

    @classmethod
    def write_json(cls, obj, params):
        if params.data_dir is not None:
            path = params.data_dir / f"{id(obj)}.bin"
            path.write_bytes(obj)
            return path.name
        else:
            return bytes_to_base64(obj, True)

    @classmethod
    def validate(cls, obj, criteria):
        return obj


class _(Serializer[PILImage]):
    @classmethod
    def read_json(cls, obj, params):
        if not isinstance(obj, str):
            raise ValueError

        if params.data_dir is not None:
            return load_image(params.data_dir / obj)
        else:
            return np_to_pil(base64_to_image(obj, True))

    @classmethod
    def write_json(cls, obj, params):
        if params.data_dir is not None:
            path = params.data_dir / f"{id(obj)}.png"
            save_image(obj, path)
            return path.name
        else:
            return image_to_base64(pil_to_np(obj), True, "fast")

    @classmethod
    def validate(cls, obj, criteria):
        image = pil_to_np(obj)

        if (channels := criteria.get("channels")) is not None and image.shape[-1] != channels:
            image = ensure_image_dims(image, channels = criteria["channels"])

        return np_to_pil(image)


class _(Serializer[FloatArray]):
    @classmethod
    def read_json(cls, obj, params):
        if not isinstance(obj, str):
            raise ValueError

        if params.data_dir is not None:
            return load_array(params.data_dir / obj)
        else:
            return base64_to_array(obj, True)

    @classmethod
    def write_json(cls, obj, params):
        if params.data_dir is not None:
            path = params.data_dir / f"{id(obj)}.npz"
            save_array(obj, path)
            return path.name
        else:
            return array_to_base64(obj, True)

    @classmethod
    def validate(cls, obj, criteria):
        return obj


class _(Serializer[NumpyImage]):
    @classmethod
    def read_json(cls, obj, params):
        if not isinstance(obj, str):
            raise ValueError

        if params.data_dir is not None:
            return load_array(params.data_dir / obj)
        else:
            return base64_to_image(obj, True)

    @classmethod
    def write_json(cls, obj, params):
        if params.data_dir is not None:
            path = params.data_dir / f"{id(obj)}.npz"
            save_array(obj, path)
            return path.name
        else:
            return image_to_base64(obj, True, "fast")

    @classmethod
    def validate(cls, obj, criteria):
        if (channels := criteria.get("channels")) is not None and obj.shape[-1] != channels:
            obj = ensure_image_dims(obj, channels = criteria["channels"])

        return obj
