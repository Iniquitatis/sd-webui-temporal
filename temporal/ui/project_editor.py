from typing import Iterator

from temporal.pipeline import Pipeline
from temporal.pipeline_module import PIPELINE_MODULES
from temporal.project import Project
from temporal.ui import CallbackInputs, CallbackOutputs, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.animation_editor import AnimationEditor
from temporal.ui.initial_noise_editor import InitialNoiseEditor
from temporal.ui.pipeline_editor import PipelineEditor


class ProjectEditor(Widget):
    def __init__(
        self,
        value: Project = Project(),
    ) -> None:
        super().__init__()

        value.pipeline = Pipeline(modules = [cls() for cls in sorted(PIPELINE_MODULES, key = lambda x: f"{x.icon} {x.name}")])

        self._initial_noise = InitialNoiseEditor(value.initial_noise)
        self._pipeline = PipelineEditor(value.pipeline)
        self._animation = AnimationEditor(value.animation)

        @self._pipeline.callback("change", [self._pipeline], [self._animation])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            pipeline = inputs[self]

            # TODO: Read properties here
            properties = {
                "parameters.strength": "number",
                "pipeline.modules[0].enabled": "bool",
                "pipeline.modules[0].amount": "number",
                "pipeline.modules[1].enabled": "bool",
                "pipeline.modules[1].amount": "number",
                "pipeline.modules[2].enabled": "bool",
                "pipeline.modules[2].amount": "number",
                "pipeline.modules[2].color": "color",
            }

            return {self._animation: {"properties": properties}}

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._initial_noise
        yield self._pipeline
        yield self._animation

    def read(self, data: ReadData) -> Project:
        return Project(
            initial_noise = data[self._initial_noise],
            pipeline = data[self._pipeline],
            animation = data[self._animation],
        )

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {
            self._initial_noise: {},
            self._pipeline: {},
            self._animation: {},
        }

        if isinstance(value := data.get("value", None), Project):
            result[self._initial_noise]["value"] = value.initial_noise
            result[self._pipeline]["value"] = value.pipeline
            result[self._animation]["value"] = value.animation

        if isinstance(preview_states := data.get("preview_states", None), dict):
            result[self._pipeline]["preview_states"] = preview_states

        return result
