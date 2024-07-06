from typing import Iterator

import gradio as gr

from temporal.pipeline import Pipeline
from temporal.ui import ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.reorderable_list import ReorderableList
from temporal.ui.pipeline_module_editor import PipelineModuleEditor
from temporal.utils.collection import find_by_predicate, reorder_dict


class PipelineEditor(Widget):
    def __init__(
        self,
        value: Pipeline = Pipeline(),
    ) -> None:
        super().__init__()

        self._parallel = GradioWidget(gr.Number, label = "Parallel", precision = 0, minimum = 1, step = 1, value = value.parallel)

        with ReorderableList() as self._module_order:
            self._modules = {
                module.id: PipelineModuleEditor(value = module)
                for module in value.modules
            }

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._parallel
        yield self._module_order
        yield from self._modules.values()

    def read(self, data: ReadData) -> Pipeline:
        return Pipeline(
            parallel = data[self._parallel],
            modules = [
                data[widget]
                for widget in reorder_dict(self._modules, [
                    self._module_ids[x]
                    for x in data[self._module_order]
                ]).values()
            ],
        )

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {
            self._parallel: {},
            self._module_order: {},
        }
        result |= {widget: {} for widget in self._modules.values()}

        if isinstance(value := data.get("value", None), Pipeline):
            result[self._parallel]["value"] = value.parallel
            result[self._module_order]["value"] = [self._module_ids.index(x.id) for x in value.modules]

            for id, widget in self._modules.items():
                result[widget]["value"] = find_by_predicate(value.modules, lambda x: x.id == id)

        if isinstance(preview_states := data.get("preview_states", None), dict):
            for id, widget in self._modules.items():
                result[widget]["preview"] = preview_states[id]

        return result

    @property
    def _module_ids(self) -> list[str]:
        return list(self._modules.keys())
