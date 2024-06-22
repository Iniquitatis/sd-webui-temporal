from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Type, TypeVar, cast

import gradio as gr
import numpy as np

from temporal.color import Color
from temporal.meta.configurable import BoolParam, ColorParam, ConfigurableParam, EnumParam, FloatParam, ImageParam, IntParam, NoiseParam, PathParam, StringParam
from temporal.noise import Noise
from temporal.ui import ReadData, ResolvedCallback, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.color_editor import ColorEditor
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.noise_editor import NoiseEditor
from temporal.ui.path_editor import PathEditor


T = TypeVar("T")
U = TypeVar("U")


class ConfigurableParamEditor(Widget):
    def __init__(
        self,
        param: ConfigurableParam[T],
        value: Optional[T | Callable[[], T]] = None,
    ) -> None:
        super().__init__()

        def make_static_value(type: Type[U], param: ConfigurableParam[U]) -> U:
            return value if isinstance(value, type) else param.default

        def make_dynamic_value(type: Type[U], param: ConfigurableParam[U]) -> U | Callable[[], U]:
            return cast(Callable[[], U], value) if callable(value) else make_static_value(type, param)

        if isinstance(param, BoolParam):
            self._widget = GradioWidget(gr.Checkbox, label = self._format_label(param.name), value = make_dynamic_value(bool, param))

        elif isinstance(param, IntParam) and param.ui_type == "box":
            self._widget = GradioWidget(gr.Number, label = self._format_label(param.name), precision = 0, minimum = param.minimum, maximum = param.maximum, step = param.step, value = make_dynamic_value(int, param))

        elif isinstance(param, IntParam) and param.ui_type == "slider":
            self._widget = GradioWidget(gr.Slider, label = self._format_label(param.name), minimum = param.minimum if param.minimum is not None else 0, maximum = param.maximum if param.maximum is not None else 1, step = param.step, value = make_dynamic_value(int, param))

        elif isinstance(param, FloatParam) and param.ui_type == "box":
            self._widget = GradioWidget(gr.Number, label = self._format_label(param.name), minimum = param.minimum, maximum = param.maximum, step = param.step, value = make_dynamic_value(float, param))

        elif isinstance(param, FloatParam) and param.ui_type == "slider":
            self._widget = GradioWidget(gr.Slider, label = self._format_label(param.name), minimum = param.minimum if param.minimum is not None else 0, maximum = param.maximum if param.maximum is not None else 1, step = param.step, value = make_dynamic_value(float, param))

        elif isinstance(param, StringParam) and param.ui_type == "box":
            self._widget = GradioWidget(gr.Textbox, label = self._format_label(param.name), value = make_dynamic_value(str, param))

        elif isinstance(param, StringParam) and param.ui_type == "area":
            self._widget = GradioWidget(gr.TextArea, label = self._format_label(param.name), value = make_dynamic_value(str, param))

        elif isinstance(param, StringParam) and param.ui_type == "code":
            self._widget = GradioWidget(gr.Code, label = self._format_label(param.name), language = cast(None, param.language), value = make_dynamic_value(str, param))

        elif isinstance(param, PathParam):
            self._widget = PathEditor(label = self._format_label(param.name), value = make_dynamic_value(Path, param))

        elif isinstance(param, EnumParam) and param.ui_type == "menu":
            self._widget = GradioWidget(gr.Dropdown, label = self._format_label(param.name), choices = list(param.choices), value = make_dynamic_value(str, param))

        elif isinstance(param, EnumParam) and param.ui_type == "radio":
            self._widget = GradioWidget(gr.Radio, label = self._format_label(param.name), choices = list(param.choices), value = make_dynamic_value(str, param))

        elif isinstance(param, ColorParam):
            self._widget = ColorEditor(label = self._format_label(param.name), channels = param.channels, value = make_static_value(Color, param))

        elif isinstance(param, ImageParam):
            self._widget = GradioWidget(gr.Image, label = self._format_label(param.name), type = "numpy", image_mode = "RGBA" if param.channels == 4 else "RGB", value = make_static_value(np.ndarray, param))

        elif isinstance(param, NoiseParam):
            self._widget = NoiseEditor(label = self._format_label(param.name), value = make_static_value(Noise, param))

        else:
            raise NotImplementedError

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield from self._widget.dependencies

    def read(self, data: ReadData) -> Any:
        return self._widget.read(data)

    def update(self, data: UpdateData) -> UpdateRequest:
        return self._widget.update(data)

    def setup_callback(self, callback: ResolvedCallback) -> None:
        self._widget.setup_callback(callback)
