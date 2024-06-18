from pathlib import Path
from typing import Any, Generic, Literal, Optional, TypeVar, cast

from temporal.meta.registerable import Registerable
from temporal.meta.serializable import Serializable, SerializableField
from temporal.utils.image import NumpyImage


T = TypeVar("T")


class ConfigurableParam(SerializableField, Generic[T]):
    def __new__(cls, *args: Any, **kwargs: Any) -> T:
        instance = object.__new__(cls)
        instance.__init__(*args, **kwargs)
        return cast(T, instance)

    def __init__(self, name: str = "Parameter", default: Optional[T] = None) -> None:
        super().__init__(default = default)
        self.name = name


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


#===============================================================================


class BoolParam(ConfigurableParam[bool]):
    def __init__(
        self,
        name: str = "Parameter",
        value: bool = False,
    ) -> None:
        super().__init__(name, value)


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


class PathParam(ConfigurableParam[Path]):
    def __init__(
        self,
        name: str = "Parameter",
        value: Path = Path(),
    ) -> None:
        super().__init__(name, value)


class EnumParam(ConfigurableParam[str]):
    def __init__(
        self,
        name: str = "Parameter",
        choices: list[str] = [],
        value: str = "",
        ui_type: Literal["menu", "radio"] = "menu",
    ) -> None:
        super().__init__(name, value)
        self.choices = choices
        self.ui_type = ui_type


class ColorParam(ConfigurableParam[str]):
    def __init__(
        self,
        name: str = "Parameter",
        channels: int = 3,
        value: str = "#ffffff",
    ) -> None:
        super().__init__(name, value)
        self.channels = channels


class ImageParam(ConfigurableParam[NumpyImage]):
    def __init__(
        self,
        name: str = "Parameter",
        channels: int = 3,
    ) -> None:
        super().__init__(name, None)
        self.channels = channels
