from typing import Iterator

import gradio as gr
import numpy as np

from temporal.gradient import Gradient
from temporal.ui import Callback, CallbackInputs, CallbackOutputs, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.color_editor import ColorEditor
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.radio import Radio
from temporal.utils.image import NumpyImage, alpha_blend, checkerboard


class GradientEditor(Widget):
    def __init__(
        self,
        label: str = "",
        value: Gradient = Gradient(),
        visible: bool = True,
    ) -> None:
        super().__init__()

        with GradioWidget(gr.Row, visible = visible, elem_classes = ["temporal-gap"]) as self._row:
            self._preview = GradioWidget(gr.Image, label = self._format_label(label, "Preview"), type = "numpy", image_mode = "RGB", value = self._generate_preview_texture(value))

            with GradioWidget(gr.Column):
                self._type = Radio(label = self._format_label(label, "Type"), choices = [("linear", "Linear"), ("radial", "Radial")], value = value.type)

                with GradioWidget(gr.Row):
                    self._start_x = GradioWidget(gr.Number, label = self._format_label(label, "Start X"), step = 0.01, value = value.start_x)
                    self._start_y = GradioWidget(gr.Number, label = self._format_label(label, "Start Y"), step = 0.01, value = value.start_y)

                with GradioWidget(gr.Row):
                    self._end_x = GradioWidget(gr.Number, label = self._format_label(label, "End X"), step = 0.01, value = value.end_x)
                    self._end_y = GradioWidget(gr.Number, label = self._format_label(label, "End Y"), step = 0.01, value = value.end_y)

                self._start_color = ColorEditor(label = self._format_label(label, "Start color"), value = value.start_color)
                self._end_color = ColorEditor(label = self._format_label(label, "End color"), value = value.end_color)

        @self.callback("change", [self], [self._preview])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            return {self._preview: {"value": self._generate_preview_texture(inputs[self])}}

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._type
        yield self._start_x
        yield self._start_y
        yield self._end_x
        yield self._end_y
        yield self._start_color
        yield self._end_color

    def read(self, data: ReadData) -> Gradient:
        return Gradient(
            data[self._type],
            data[self._start_x],
            data[self._start_y],
            data[self._end_x],
            data[self._end_y],
            data[self._start_color],
            data[self._end_color],
        )

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(visible := data.get("visible", None), bool):
            result[self._row] = {"visible": visible}

        if isinstance(value := data.get("value", None), Gradient):
            result[self._type] = {"value": value.type}
            result[self._start_x] = {"value": value.start_x}
            result[self._start_y] = {"value": value.start_y}
            result[self._end_x] = {"value": value.end_x}
            result[self._end_y] = {"value": value.end_y}
            result[self._start_color] = {"value": value.start_color}
            result[self._end_color] = {"value": value.end_color}

        return result

    def setup_callback(self, callback: Callback) -> None:
        self._type.setup_callback(callback)
        self._start_x.setup_callback(callback)
        self._start_y.setup_callback(callback)
        self._end_x.setup_callback(callback)
        self._end_y.setup_callback(callback)
        self._start_color.setup_callback(callback)
        self._end_color.setup_callback(callback)

    def _generate_preview_texture(self, gradient: Gradient) -> NumpyImage:
        return alpha_blend(
            checkerboard((256, 256, 3), 8, np.array([0.75, 0.75, 0.75]), np.array([0.25, 0.25, 0.25])),
            gradient.generate((256, 256, 4), True),
        )
