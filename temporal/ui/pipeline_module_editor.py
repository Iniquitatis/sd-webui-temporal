from typing import Iterator

import gradio as gr

from temporal.blend_modes import BLEND_MODES
from temporal.image_filters import ImageFilter
from temporal.pipeline_modules import PipelineModule
from temporal.shared import shared
from temporal.ui import CallbackInputs, CallbackOutputs, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.configurable_param_editor import ConfigurableParamEditor
from temporal.ui.dropdown import Dropdown
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.image_mask_editor import ImageMaskEditor
from temporal.ui.module_list import ModuleAccordion, ModuleAccordionSpecialCheckbox


class PipelineModuleEditor(Widget):
    def __init__(
        self,
        value: PipelineModule = PipelineModule(),
    ) -> None:
        super().__init__()

        self.type = value.__class__

        with ModuleAccordion(label = f"{value.icon} {value.name}", key = value.id, value = value.enabled, open = False) as self._enabled:
            self._visibility = ModuleAccordionSpecialCheckbox(value = lambda: shared.previewed_modules[value.id], classes = ["temporal-visibility-checkbox"])

            @self._visibility.callback("change", [self._visibility], [])
            def _(inputs: CallbackInputs) -> CallbackOutputs:
                shared.previewed_modules[value.id] = inputs[self._visibility]
                return {}

            if isinstance(value, ImageFilter):
                with GradioWidget(gr.Row):
                    self._amount = GradioWidget(gr.Slider, label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = value.amount)
                    self._amount_relative = GradioWidget(gr.Checkbox, label = "Relative", value = value.amount_relative)

                self._blend_mode = Dropdown(label = "Blend mode", choices = [(x.id, x.name) for x in BLEND_MODES.values()], value = value.blend_mode)

                with GradioWidget(gr.Tab, label = "Parameters"):
                    self._params = {
                        key: ConfigurableParamEditor(param = param, value = getattr(value, key))
                        for key, param in value.__params__.items()
                    }

                with GradioWidget(gr.Tab, label = "Mask"):
                    self._mask = ImageMaskEditor(value = value.mask)

            else:
                self._params = {
                    key: ConfigurableParamEditor(param = param, value = getattr(value, key))
                    for key, param in value.__params__.items()
                }

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._enabled
        yield from self._params.values()

        if issubclass(self.type, ImageFilter):
            yield self._amount
            yield self._amount_relative
            yield self._blend_mode
            yield self._mask

    def read(self, data: ReadData) -> PipelineModule:
        kwargs = {}
        kwargs["enabled"] = data[self._enabled]
        kwargs |= {key: data[widget] for key, widget in self._params.items()}

        if issubclass(self.type, ImageFilter):
            kwargs["amount"] = data[self._amount]
            kwargs["amount_relative"] = data[self._amount_relative]
            kwargs["blend_mode"] = data[self._blend_mode]
            kwargs["mask"] = data[self._mask]

        return self.type(**kwargs)

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), PipelineModule):
            result[self._enabled] = {"value": value.enabled}
            result |= {widget: {"value": getattr(value, key)} for key, widget in self._params.items()}

            if isinstance(value, ImageFilter):
                result[self._amount] = {"value": value.amount}
                result[self._amount_relative] = {"value": value.amount_relative}
                result[self._blend_mode] = {"value": value.blend_mode}
                result[self._mask] = {"value": value.mask}

        return result
