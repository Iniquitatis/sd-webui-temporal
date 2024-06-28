from typing import Any, Callable, Iterator

import gradio as gr

from temporal.global_options import OptionCategory
from temporal.ui import ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.configurable_param_editor import ConfigurableParamEditor
from temporal.ui.gradio_widget import GradioWidget


class OptionCategoryEditor(Widget):
    def __init__(
        self,
        value: OptionCategory = OptionCategory(),
    ) -> None:
        super().__init__()

        self.type = value.__class__

        with GradioWidget(gr.Accordion, label = value.name, open = False):
            def make_getter(key: str) -> Callable[[], Any]:
                return lambda: getattr(value, key)

            self._params = {
                key: ConfigurableParamEditor(param = param, value = make_getter(key))
                for key, param in value.__params__.items()
            }

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield from self._params.values()

    def read(self, data: ReadData) -> OptionCategory:
        return self.type(**{key: data[widget] for key, widget in self._params.items()})

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), OptionCategory):
            result |= {widget: {"value": getattr(value, key)} for key, widget in self._params.items()}

        return result
