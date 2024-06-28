from typing import Iterator

import gradio as gr

from temporal.pipeline import Pipeline
from temporal.ui import ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.module_list import ModuleList
from temporal.ui.pipeline_module_editor import PipelineModuleEditor


class PipelineEditor(Widget):
    def __init__(
        self,
        value: Pipeline = Pipeline(),
    ) -> None:
        super().__init__()

        self._parallel = GradioWidget(gr.Number, label = "Parallel", precision = 0, minimum = 1, step = 1, value = value.parallel)

        with ModuleList(keys = value.module_order) as self._module_order:
            self._modules = {
                id: PipelineModuleEditor(value = module)
                for id, module in value.modules.items()
            }

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._parallel
        yield self._module_order
        yield from self._modules.values()

    def read(self, data: ReadData) -> Pipeline:
        return Pipeline(
            parallel = data[self._parallel],
            module_order = data[self._module_order],
            modules = {key: data[widget] for key, widget in self._modules.items()},
        )

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), Pipeline):
            result[self._parallel] = {"value": value.parallel}
            result[self._module_order] = {"value": value.module_order}
            result |= {widget: {"value": value.modules[key]} for key, widget in self._modules.items()}

        return result
