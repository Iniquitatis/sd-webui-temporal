from typing import Iterator

import gradio as gr

from temporal.color import Color
from temporal.ui import Callback, CallbackInputs, CallbackOutputs, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget


class ColorEditor(Widget):
    def __init__(
        self,
        label: str = "",
        channels: int = 4,
        step: float = 0.01,
        value: Color = Color(),
    ) -> None:
        super().__init__()

        with GradioWidget(gr.Group):
            GradioWidget(gr.HTML, f"<span class='temporal-composite-widget-title'>{label}</span>")

            with GradioWidget(gr.Row):
                with GradioWidget(gr.Column):
                    self._r = GradioWidget(gr.Slider, label = self._format_label("R"), minimum = 0.0, maximum = 1.0, step = step, value = value.r, visible = channels >= 1, elem_classes = ["temporal-color-editor-slider"])
                    self._g = GradioWidget(gr.Slider, label = self._format_label("G"), minimum = 0.0, maximum = 1.0, step = step, value = value.g, visible = channels >= 2, elem_classes = ["temporal-color-editor-slider"])
                    self._b = GradioWidget(gr.Slider, label = self._format_label("B"), minimum = 0.0, maximum = 1.0, step = step, value = value.b, visible = channels >= 3, elem_classes = ["temporal-color-editor-slider"])
                    self._a = GradioWidget(gr.Slider, label = self._format_label("A"), minimum = 0.0, maximum = 1.0, step = step, value = value.a, visible = channels >= 4, elem_classes = ["temporal-color-editor-slider"])

                self._preview = GradioWidget(gr.ColorPicker, label = self._format_label("Preview"), interactive = False, value = value.to_hex(3), elem_classes = ["temporal-color-editor-picker"])

        @self.callback("change", [self], [self._preview])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            return {self._preview: {"value": inputs[self].to_hex(3)}}

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._r
        yield self._g
        yield self._b
        yield self._a

    def read(self, data: ReadData) -> Color:
        return Color(
            float(data[self._r]),
            float(data[self._g]),
            float(data[self._b]),
            float(data[self._a]),
        )

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {
            self._r: {},
            self._g: {},
            self._b: {},
            self._a: {},
        }

        if isinstance(channels := data.get("channels", None), int):
            result[self._r]["visible"] = channels >= 1
            result[self._g]["visible"] = channels >= 2
            result[self._b]["visible"] = channels >= 3
            result[self._a]["visible"] = channels >= 4

        if isinstance(value := data.get("value", None), Color):
            result[self._r]["value"] = value.r
            result[self._g]["value"] = value.g
            result[self._b]["value"] = value.b
            result[self._a]["value"] = value.a

        return result

    def setup_callback(self, callback: Callback) -> None:
        self._r.setup_callback(callback)
        self._g.setup_callback(callback)
        self._b.setup_callback(callback)
        self._a.setup_callback(callback)
