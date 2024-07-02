from random import randint
from typing import Iterator

import gradio as gr

from modules.ui_components import ToolButton

from temporal.ui import Callback, CallbackInputs, CallbackOutputs, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget


class SeedEditor(Widget):
    def __init__(
        self,
        label: str = "",
        value: int = 0,
    ) -> None:
        super().__init__()

        with GradioWidget(gr.Row):
            self._value = GradioWidget(gr.Number, label = self._format_label(label), precision = 0, minimum = 0, step = 1, value = value)
            self._randomize = GradioWidget(ToolButton, value = "\U0001f3b2\ufe0f")

        @self._randomize.callback("click", [], [self._value])
        def _(_: CallbackInputs) -> CallbackOutputs:
            return {self._value: {"value": randint(0, 2 ** 32 - 1)}}

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield from self._value.dependencies

    def read(self, data: ReadData) -> int:
        return self._value.read(data)

    def update(self, data: UpdateData) -> UpdateRequest:
        return self._value.update(data)

    def setup_callback(self, callback: Callback) -> None:
        self._value.setup_callback(callback)
