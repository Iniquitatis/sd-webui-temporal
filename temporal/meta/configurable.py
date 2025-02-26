from typing import Any, Callable, Literal, Optional, TypeVar, cast

from temporal.meta.registerable import Registerable
from temporal.meta.serializable import Serializable, SerializableField
from temporal.serialization import SerializationParams, serialize
from temporal.utils.typing import get_full_type_name


T = TypeVar("T")


class ConfigurableParam(SerializableField[T]):
    def __new__(cls, *args: Any, **kwargs: Any) -> T:
        instance = object.__new__(cls)
        instance.__init__(*args, **kwargs)
        return cast(T, instance)

    def __init__(
        self,
        name: str = "Parameter",
        *,
        value: Optional[T] = None,
        factory: Optional[Callable[[], T]] = None,
        variant: str = "",
        minimum: Optional[int | float] = None,
        maximum: Optional[int | float] = None,
        step: Optional[int | float] = None,
        axes: Optional[list[str]] = None,
        channels: Optional[int] = None,
        choices: Optional[list[str | tuple[Any, str]]] = None,
        language: Optional[str] = None,
        ui_type: Optional[Literal["area", "box", "code", "menu", "radio", "slider"]] = None,
    ) -> None:
        super().__init__(value = value, factory = factory, variant = variant)
        self.name = name
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.axes = axes
        self.channels = channels
        self.choices = {
            x[0] if isinstance(x, tuple) else x:
            x[1] if isinstance(x, tuple) else x
            for x in choices
        } if choices else None
        self.language = language
        self.ui_type = ui_type

    @property
    def schema(self) -> dict[str, Any]:
        print({"type": get_full_type_name(self.type),
            "name": self.name,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "axes": self.axes,
            "channels": self.channels,
            "choices": self.choices,
            "language": self.language,
            "ui_type": self.ui_type,
            "default": self.default,
        })

        return {
            "type": get_full_type_name(self.type),
            "name": self.name,
            **({"minimum": self.minimum} if self.minimum is not None else {}),
            **({"maximum": self.maximum} if self.maximum is not None else {}),
            **({"step": self.step} if self.step is not None else {}),
            **({"axes": self.axes} if self.axes is not None else {}),
            **({"channels": self.channels} if self.channels is not None else {}),
            **({"choices": self.choices} if self.choices is not None else {}),
            **({"language": self.language} if self.language is not None else {}),
            **({"ui_type": self.ui_type} if self.ui_type is not None else {}),
            **({"default": serialize(self.type, self.variant, self.default, SerializationParams())} if self.default is not None else {}),
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
            "type": cls.id,
            "name": cls.name,
            "parameters": {
                key: param.schema
                for key, param in cls.__params__.items()
            },
        }
