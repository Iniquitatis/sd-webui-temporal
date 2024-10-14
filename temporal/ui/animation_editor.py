from typing import Iterator

import gradio as gr

from temporal.animation import Animation
from temporal.animation.parsing import parse_animation
from temporal.animation.plotting import plot_animation
from temporal.animation.printing import print_animation
from temporal.ui import CallbackInputs, CallbackOutputs, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget


class AnimationEditor(Widget):
    def __init__(
        self,
        value: Animation | str = Animation(),
    ) -> None:
        super().__init__()

        self._code = GradioWidget(gr.Code, label = "Animation", language = "python", value = print_animation(value) if isinstance(value, Animation) else value)
        self._render_graphs = GradioWidget(gr.Button, value = "Render graphs")
        self._graphs = GradioWidget(gr.Gallery, label = "Graphs", columns = 4, object_fit = "contain", preview = True)

        @self._render_graphs.callback("click", [self], [self._graphs])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            return {self._graphs: {"value": plot_animation(inputs[self])}}

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._code

    def read(self, data: ReadData) -> Animation:
        return parse_animation(data[self._code])

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if (value := data.get("value", None)) is not None:
            result[self._code] = {"value": print_animation(value) if isinstance(value, Animation) else value}

        return result
