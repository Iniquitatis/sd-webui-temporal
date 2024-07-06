from typing import Iterator

import gradio as gr

from temporal.blend_modes import BLEND_MODES, BlendMode, NormalBlendMode
from temporal.ui import Callback, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.dropdown import Dropdown


class BlendModeSelector(Widget):
    def __init__(
        self,
        label: str = "",
        value: BlendMode = NormalBlendMode(),
    ) -> None:
        super().__init__()

        self._selector = Dropdown(label = label, choices = [(x, x.name) for x in BLEND_MODES], value = value.__class__)

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._selector

    def read(self, data: ReadData) -> BlendMode:
        return data[self._selector]()

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), BlendMode):
            result[self._selector] = {"value": value.__class__}

        return result

    def setup_callback(self, callback: Callback) -> None:
        self._selector.setup_callback(callback)
