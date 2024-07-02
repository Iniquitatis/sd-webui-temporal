from typing import Iterator

import gradio as gr

from temporal.project import InitialNoiseParams
from temporal.ui import ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.noise_editor import NoiseEditor


class InitialNoiseEditor(Widget):
    def __init__(
        self,
        value: InitialNoiseParams = InitialNoiseParams(),
    ) -> None:
        super().__init__()

        with GradioWidget(gr.Accordion, label = "Initial noise", open = False):
            self._factor = GradioWidget(gr.Slider, label = "Factor", minimum = 0.0, maximum = 1.0, step = 0.01, value = value.factor)
            self._noise = NoiseEditor(value = value.noise)

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._factor
        yield self._noise

    def read(self, data: ReadData) -> InitialNoiseParams:
        return InitialNoiseParams(
            factor = data[self._factor],
            noise = data[self._noise],
        )

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), InitialNoiseParams):
            result[self._factor] = {"value": value.factor}
            result[self._noise] = {"value": value.noise}

        return result
