from typing import Any, Callable, Literal, Optional, Type, TypeVar, cast

from temporal.meta.registerable import Registerable
from temporal.meta.serializable import Serializable, SerializableField, UndefinedValue
from temporal.serialization import SerializationParams, serialize
from temporal.utils.typing import get_full_type_name


T = TypeVar("T")


Choices = list[str] | dict[T, str]


class ConfigurableParam(SerializableField[T]):
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
        ui_type: Optional[Literal["area", "box", "code", "menu", "radio", "slider"]] = None,
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
        ui_type: Optional[Literal["area", "box", "code", "menu", "radio", "slider"]] = None,
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
        self.ui_type = ui_type

    @property
    def schema(self) -> dict[str, Any]:
        choices = self.choices

        if choices is not None:
            choices = choices() if callable(choices) else choices

            if isinstance(choices, list):
                choices = {x: x for x in choices}

        return {
            "type": get_full_type_name(self.type),
            "name": self.name,
            **({"minimum": self.minimum} if self.minimum is not None else {}),
            **({"maximum": self.maximum} if self.maximum is not None else {}),
            **({"step": self.step} if self.step is not None else {}),
            **({"axes": self.axes} if self.axes is not None else {}),
            **({"channels": self.channels} if self.channels is not None else {}),
            **({"choices": choices} if choices is not None else {}),
            **({"language": self.language} if self.language is not None else {}),
            **({"ui_type": self.ui_type} if self.ui_type is not None else {}),
            **({"default": serialize(self.type, self.default, SerializationParams())} if self.default is not None else {}),
        }


class Configurable(Registerable, Serializable):
    __params__: dict[str, ConfigurableParam[Any]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__params__ = {
            key: field
            for key, field in cls.__fields__.items()
            if isinstance(field, ConfigurableParam)
        }

    @classmethod
    def schema(cls) -> dict[str, Any]:
        return {
            "type": cls.__type_name__,
            "name": cls.name,
            "parameters": {
                key: param.schema
                for key, param in cls.__params__.items()
            },
        }
