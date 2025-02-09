from pathlib import Path
from typing import Any, Callable, Generic, Literal, Optional, TypeVar, cast

from temporal.color import Color
from temporal.gradient import Gradient
from temporal.image_source import ImageSource
from temporal.meta.registerable import Registerable
from temporal.meta.serializable import Serializable, SerializableField
from temporal.noise import Noise
from temporal.pattern import Pattern
from temporal.utils.image import NumpyImage


T = TypeVar("T")


class ConfigurableParam(SerializableField, Generic[T]):
    def __new__(cls, *args: Any, **kwargs: Any) -> T:
        instance = object.__new__(cls)
        instance.__init__(*args, **kwargs)
        return cast(T, instance)

    def __init__(self, name: str = "Parameter", value: Optional[T] = None, *, factory: Optional[Callable[[], T]] = None) -> None:
        super().__init__(value = value, factory = factory)
        self.name = name

    @property
    def default(self) -> T:
        return cast(T, super().default)

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
        }


class Configurable(Registerable, Serializable):
    __params__: dict[str, ConfigurableParam[Any]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__params__ = {
            key: param
            for base_cls in list(reversed(cls.__mro__)) + [cls]
            if issubclass(base_cls, Configurable)
            for key, param in base_cls.__dict__.items()
            if isinstance(param, ConfigurableParam)
        }

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return {
            "id": cls.id,
            "name": cls.name,
            "parameters": {
                field_id: field.schema()
                for field_id, field in cls.__params__.items()
            },
        }


#===============================================================================


class BoolParam(ConfigurableParam[bool]):
    def __init__(
        self,
        name: str = "Parameter",
        value: bool = False,
    ) -> None:
        super().__init__(name, value)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "bool",
            "name": self.name,
            "default": self.default,
        }


class IntParam(ConfigurableParam[int]):
    def __init__(
        self,
        name: str = "Parameter",
        minimum: Optional[int] = None,
        maximum: Optional[int] = None,
        step: int = 1,
        value: int = 0,
        ui_type: Literal["box", "slider"] = "box",
    ) -> None:
        super().__init__(name, value)
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.ui_type = ui_type

    def schema(self) -> dict[str, Any]:
        return {
            "type": "int",
            "name": self.name,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "default": self.default,
            "ui_type": self.ui_type,
        }


class FloatParam(ConfigurableParam[float]):
    def __init__(
        self,
        name: str = "Parameter",
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        step: float = 1.0,
        value: float = 0.0,
        ui_type: Literal["box", "slider"] = "box",
    ) -> None:
        super().__init__(name, value)
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.ui_type = ui_type

    def schema(self) -> dict[str, Any]:
        return {
            "type": "float",
            "name": self.name,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "default": self.default,
            "ui_type": self.ui_type,
        }


class StringParam(ConfigurableParam[str]):
    def __init__(
        self,
        name: str = "Parameter",
        value: str = "",
        ui_type: Literal["box", "area", "code"] = "box",
        language: Optional[str] = None,
    ) -> None:
        super().__init__(name, value)
        self.ui_type = ui_type
        self.language = language

    def schema(self) -> dict[str, Any]:
        return {
            "type": "string",
            "name": self.name,
            "default": self.default,
            "ui_type": self.ui_type,
            "language": self.language,
        }


class PathParam(ConfigurableParam[Path]):
    def __init__(
        self,
        name: str = "Parameter",
        value: Path = Path(),
    ) -> None:
        super().__init__(name, value)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "path",
            "name": self.name,
            "default": self.default.as_posix(),
        }


class EnumParam(ConfigurableParam[str]):
    def __init__(
        self,
        name: str = "Parameter",
        choices: list[str | tuple[Any, str]] = [],
        value: str = "",
        ui_type: Literal["menu", "radio"] = "menu",
    ) -> None:
        super().__init__(name, value)
        self.choices = choices
        self.ui_type = ui_type

    def schema(self) -> dict[str, Any]:
        return {
            "type": "enum",
            "name": self.name,
            "choices": {
                x[0] if isinstance(x, tuple) else x:
                x[1] if isinstance(x, tuple) else x
                for x in self.choices
            },
            "default": self.default,
            "ui_type": self.ui_type,
        }


class ColorParam(ConfigurableParam[Color]):
    def __init__(
        self,
        name: str = "Parameter",
        channels: int = 3,
        factory: Callable[[], Color] = Color,
    ) -> None:
        super().__init__(name, factory = factory)
        self.channels = channels

    def schema(self) -> dict[str, Any]:
        return {
            "type": "color",
            "name": self.name,
            "channels": self.channels,
            "default": self.default.to_hex(self.channels),
        }


class ImageParam(ConfigurableParam[NumpyImage]):
    def __init__(
        self,
        name: str = "Parameter",
        channels: int = 3,
    ) -> None:
        super().__init__(name, None)
        self.channels = channels

    def schema(self) -> dict[str, Any]:
        return {
            "type": "image",
            "name": self.name,
            "channels": self.channels,
        }


class ImageSourceParam(ConfigurableParam[ImageSource]):
    def __init__(
        self,
        name: str = "Parameter",
        channels: int = 3,
        factory: Callable[[], ImageSource] = ImageSource,
    ) -> None:
        super().__init__(name, factory = factory)
        self.channels = channels

    def schema(self) -> dict[str, Any]:
        return {
            "type": "image_source",
            "name": self.name,
            "default": self.default.to_json(),
        }


class GradientParam(ConfigurableParam[Gradient]):
    def __init__(
        self,
        name: str = "Parameter",
        factory: Callable[[], Gradient] = Gradient,
    ) -> None:
        super().__init__(name, factory = factory)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "gradient",
            "name": self.name,
            "default": self.default.to_json(),
        }


class NoiseParam(ConfigurableParam[Noise]):
    def __init__(
        self,
        name: str = "Parameter",
        factory: Callable[[], Noise] = Noise,
    ) -> None:
        super().__init__(name, factory = factory)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "noise",
            "name": self.name,
            "default": self.default.to_json(),
        }


class PatternParam(ConfigurableParam[Pattern]):
    def __init__(
        self,
        name: str = "Parameter",
        factory: Callable[[], Pattern] = Pattern,
    ) -> None:
        super().__init__(name, factory = factory)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "pattern",
            "name": self.name,
            "default": self.default.to_json(),
        }
