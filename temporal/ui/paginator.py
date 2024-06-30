from inspect import isgeneratorfunction
from typing import Iterator, Optional, cast

import gradio as gr

from modules.ui_components import ToolButton

from temporal.ui import Callback, CallbackFunc, CallbackInputs, CallbackOutputs, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget


class Paginator(Widget):
    def __init__(
        self,
        label: str = "",
        minimum: Optional[int] = None,
        maximum: Optional[int] = None,
        value: int = 0,
    ) -> None:
        super().__init__()

        with GradioWidget(gr.Row):
            self._previous_button = GradioWidget(ToolButton, value = "<")
            self._index = GradioWidget(gr.Number, label = self._format_label(label), precision = 0, minimum = minimum, maximum = maximum, step = 1, value = value)
            self._next_button = GradioWidget(ToolButton, value = ">")

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._index

    def read(self, data: ReadData) -> int:
        return data[self._index]

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), int):
            result[self._index] = {"value": value}

        return result

    def setup_callback(self, callback: Callback) -> None:
        if callback.event != "change" or isgeneratorfunction(callback.func):
            raise ValueError

        func = cast(CallbackFunc, callback.func)

        def previous_page(inputs: CallbackInputs) -> CallbackOutputs:
            inputs[self] = self._clamp(inputs.pop(self) - 1)
            return {self: {"value": inputs[self]}} | func(inputs)

        def set_page(inputs: CallbackInputs) -> CallbackOutputs:
            inputs[self] = self._clamp(inputs.pop(self))
            return {self: {"value": inputs[self]}} | func(inputs)

        def next_page(inputs: CallbackInputs) -> CallbackOutputs:
            inputs[self] = self._clamp(inputs.pop(self) + 1)
            return {self: {"value": inputs[self]}} | func(inputs)

        self._previous_button.setup_callback(Callback("click", previous_page, [self] + callback.inputs, [self] + callback.outputs))
        self._index.setup_callback(Callback("change", set_page, [self] + callback.inputs, [self] + callback.outputs))
        self._next_button.setup_callback(Callback("click", next_page, [self] + callback.inputs, [self] + callback.outputs))

    def _clamp(self, value: int) -> int:
        minimum = self._index._instance.minimum if self._index._instance.minimum is not None else -1e9
        maximum = self._index._instance.maximum if self._index._instance.maximum is not None else 1e9
        return min(max(value, int(minimum)), int(maximum))
