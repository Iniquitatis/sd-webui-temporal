from typing import Any, Iterator, Type

import gradio as gr

from temporal.meta.registerable import Registerable


class UIParam:
    def __init__(self, name: str, type: Type[gr.components.Component], **kwargs: Any) -> None:
        self.id = ""
        self.name = name
        self.type = type
        self.kwargs = kwargs


class Configurable(Registerable):
    params: dict[str, UIParam] = {}

    def __init_subclass__(cls: Type["Configurable"], abstract: bool = False) -> None:
        super().__init_subclass__(abstract)

        cls.params = {id: param for id, param in _iter_all_params(cls)}

        for id, param in cls.params.items():
            param.id = id


def _iter_all_params(cls: type) -> Iterator[tuple[str, UIParam]]:
    for parent_cls in reversed(cls.__mro__):
        if issubclass(parent_cls, Configurable):
            for id, param in parent_cls.params.items():
                yield id, param

    for id, param in cls.params.items():
        yield id, param
