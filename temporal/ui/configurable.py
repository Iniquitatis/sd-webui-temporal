from pathlib import Path
from typing import Any

import gradio as gr

from temporal.meta.configurable import BoolParam, ColorParam, ConfigurableParam, EnumParam, FloatParam, ImageParam, IntParam, PathParam, StringParam
from temporal.ui import UI


def make_configurable_param_editor(ui: UI, id: str, param: ConfigurableParam[Any], groups: list[str] = []) -> None:
    if isinstance(param, BoolParam):
        ui.elem(id, gr.Checkbox, label = param.name, value = param.default, groups = groups)

    elif isinstance(param, IntParam) and param.ui_type == "box":
        ui.elem(id, gr.Number, label = param.name, precision = 0, minimum = param.minimum, maximum = param.maximum, step = param.step, value = param.default, groups = groups)

    elif isinstance(param, IntParam) and param.ui_type == "slider":
        ui.elem(id, gr.Slider, label = param.name, precision = 0, minimum = param.minimum, maximum = param.maximum, step = param.step, value = param.default, groups = groups)

    elif isinstance(param, FloatParam) and param.ui_type == "box":
        ui.elem(id, gr.Number, label = param.name, minimum = param.minimum, maximum = param.maximum, step = param.step, value = param.default, groups = groups)

    elif isinstance(param, FloatParam) and param.ui_type == "slider":
        ui.elem(id, gr.Slider, label = param.name, minimum = param.minimum, maximum = param.maximum, step = param.step, value = param.default, groups = groups)

    elif isinstance(param, StringParam) and param.ui_type == "box":
        ui.elem(id, gr.Textbox, label = param.name, value = param.default, groups = groups)

    elif isinstance(param, StringParam) and param.ui_type == "area":
        ui.elem(id, gr.TextArea, label = param.name, value = param.default, groups = groups)

    elif isinstance(param, StringParam) and param.ui_type == "code":
        ui.elem(id, gr.Code, label = param.name, language = param.language, value = param.default, groups = groups)

    elif isinstance(param, PathParam):
        ui.elem(id, gr.Textbox, label = param.name, preprocessor = Path, postprocessor = Path.as_posix, value = param.default, groups = groups)

    elif isinstance(param, EnumParam) and param.ui_type == "menu":
        ui.elem(id, gr.Dropdown, label = param.name, choices = param.choices, value = param.default, groups = groups)

    elif isinstance(param, EnumParam) and param.ui_type == "radio":
        ui.elem(id, gr.Radio, label = param.name, choices = param.choices, value = param.default, groups = groups)

    elif isinstance(param, ColorParam):
        ui.elem(id, gr.ColorPicker, label = param.name, value = param.default, groups = groups)

    elif isinstance(param, ImageParam):
        ui.elem(id, gr.Image, label = param.name, type = "numpy", image_mode = "RGBA" if param.channels == 4 else "RGB", value = param.default, groups = groups)

    else:
        raise NotImplementedError
