import json
from typing import Any, Iterator

import gradio as gr

from temporal.ui import Callback, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget


class CommunicationBox(Widget):
    def __init__(
        self,
        label: str = "",
        script: str = "",
        value: Any = None,
    ) -> None:
        super().__init__()

        self._input = GradioWidget(gr.Textbox,
            label = f"{label}: Input",
            value = json.dumps(value),
            elem_classes = ["temporal-communication-input", self.communication_class],
        )
        self._input._instance.change(
            None,
            self._input._instance,
            None,
            _js = f"(dataString) => {{let data = JSON.parse(dataString); {script}}}",
        )
        self._output = GradioWidget(gr.Textbox,
            label = f"{label}: Output",
            value = json.dumps(value),
            elem_classes = ["temporal-communication-output", self.communication_class],
        )

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._output

    @property
    def communication_class(self) -> str:
        return f"temporal-communication-{self.index}"

    def read(self, data: ReadData) -> Any:
        return json.loads(data[self._output])

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if (value := data.get("value", None)) is not None:
            result[self._input] = {"value": json.dumps(value)}

        return result

    # TODO: Send the parsed JSON data here
    def setup_callback(self, callback: Callback) -> None:
        return super().setup_callback(callback)
