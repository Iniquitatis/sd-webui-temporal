from typing import Iterator

import gradio as gr

from temporal.color import Color
from temporal.ui import ReadData, ResolvedCallback, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget


class ColorEditor(Widget):
    def __init__(
        self,
        label: str = "",
        channels: int = 4,
        value: Color = Color(),
    ) -> None:
        super().__init__()

        with GradioWidget(gr.Row):
            self._color = GradioWidget(gr.ColorPicker, label = self._format_label(label, "RGB"), value = value.to_hex(3))
            self._alpha = GradioWidget(gr.Slider, label = self._format_label(label, "Alpha"), minimum = 0.0, maximum = 1.0, step = 0.01, value = value.a, visible = channels == 4)

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._color
        yield self._alpha

    def read(self, data: ReadData) -> Color:
        result = Color.from_hex(data[self._color])
        result.a = data[self._alpha]
        return result

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {
            self._color: {},
            self._alpha: {},
        }

        if isinstance(channels := data.get("channels", None), int):
            result[self._alpha]["visible"] = channels == 4

        if isinstance(value := data.get("value", None), Color):
            result[self._color]["value"] = value.to_hex(3)
            result[self._alpha]["value"] = value.a

        return result

    def setup_callback(self, callback: ResolvedCallback) -> None:
        self._color.setup_callback(callback)
        self._alpha.setup_callback(callback)
