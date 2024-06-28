from typing import Iterator

import gradio as gr

from temporal.session import InitialNoiseParams
from temporal.ui import ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.noise_editor import NoiseEditor


class InitialNoiseEditor(Widget):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        with GradioWidget(gr.Accordion, label = "Initial noise", open = False):
            self._factor = GradioWidget(gr.Slider, label = "Factor", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0)
            self._noise = NoiseEditor()
            self._use_initial_seed = GradioWidget(gr.Checkbox, label = "Use initial seed", value = False)

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._factor
        yield self._noise
        yield self._use_initial_seed

    def read(self, data: ReadData) -> InitialNoiseParams:
        return InitialNoiseParams(
            factor = data[self._factor],
            noise = data[self._noise],
            use_initial_seed = data[self._use_initial_seed],
        )

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), InitialNoiseParams):
            result[self._factor] = {"value": value.factor}
            result[self._noise] = {"value": value.noise}
            result[self._use_initial_seed] = {"value": value.use_initial_seed}

        return result
