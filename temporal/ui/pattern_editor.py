from typing import Iterator

import gradio as gr

from temporal.color import Color
from temporal.pattern import Pattern
from temporal.ui import Callback, CallbackInputs, CallbackOutputs, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.color_editor import ColorEditor
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.radio import Radio
from temporal.utils.image import NumpyImage, alpha_blend


class PatternEditor(Widget):
    def __init__(
        self,
        label: str = "",
        value: Pattern = Pattern(),
        visible: bool = True,
    ) -> None:
        super().__init__()

        with GradioWidget(gr.Row, visible = visible, elem_classes = ["temporal-gap"]) as self._row:
            self._preview = GradioWidget(gr.Image, label = self._format_label(label, "Preview"), type = "numpy", image_mode = "RGB", value = self._generate_preview_texture(value))

            with GradioWidget(gr.Column):
                self._type = Radio(label = self._format_label(label, "Type"), choices = [("horizontal_lines", "Horizontal lines"), ("vertical_lines", "Vertical lines"), ("diagonal_lines_nw", "Diagonal lines NW"), ("diagonal_lines_ne", "Diagonal lines NE"), ("checkerboard", "Checkerboard")], value = value.type)
                self._size = GradioWidget(gr.Number, label = self._format_label(label, "Size"), precision = 0, minimum = 1, step = 1, value = value.size)
                self._color_a = ColorEditor(label = self._format_label(label, "Color A"), value = value.color_a)
                self._color_b = ColorEditor(label = self._format_label(label, "Color B"), value = value.color_b)

        @self.callback("change", [self], [self._preview])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            return {self._preview: {"value": self._generate_preview_texture(inputs[self])}}

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._type
        yield self._size
        yield self._color_a
        yield self._color_b

    def read(self, data: ReadData) -> Pattern:
        return Pattern(
            data[self._type],
            data[self._size],
            data[self._color_a],
            data[self._color_b],
        )

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(visible := data.get("visible", None), bool):
            result[self._row] = {"visible": visible}

        if isinstance(value := data.get("value", None), Pattern):
            result[self._type] = {"value": value.type}
            result[self._size] = {"value": value.size}
            result[self._color_a] = {"value": value.color_a}
            result[self._color_b] = {"value": value.color_b}

        return result

    def setup_callback(self, callback: Callback) -> None:
        self._type.setup_callback(callback)
        self._size.setup_callback(callback)
        self._color_a.setup_callback(callback)
        self._color_b.setup_callback(callback)

    def _generate_preview_texture(self, pattern: Pattern) -> NumpyImage:
        return alpha_blend(
            Pattern("checkerboard", 8, Color(0.75, 0.75, 0.75), Color(0.25, 0.25, 0.25)).generate((256, 256, 3)),
            pattern.generate((256, 256, 4)),
        )
