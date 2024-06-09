from typing import Any, Type

import gradio as gr

from temporal.meta.registerable import Registerable
from temporal.meta.serializable import Serializable, SerializableField


class UIParam(SerializableField):
    def __init__(self, name: str, gr_type: Type[gr.components.Component], **kwargs: Any) -> None:
        super().__init__(default = kwargs.get("value", None))
        self.name = name
        self.gr_type = gr_type
        self.kwargs = kwargs


def ui_param(name: str, gr_type: Type[gr.components.Component], **kwargs: Any) -> Any:
    return UIParam(name, gr_type, **kwargs)


class Configurable(Registerable, Serializable):
    __ui_params__: dict[str, UIParam]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__ui_params__ = {
            key: param
            for base_cls in list(reversed(cls.__mro__)) + [cls]
            if issubclass(base_cls, Configurable)
            for key, param in base_cls.__dict__.items()
            if isinstance(param, UIParam)
        }
