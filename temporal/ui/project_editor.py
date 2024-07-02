from typing import Iterator

from temporal.project import Project
from temporal.ui import ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.initial_noise_editor import InitialNoiseEditor
from temporal.ui.pipeline_editor import PipelineEditor


class ProjectEditor(Widget):
    def __init__(
        self,
        value: Project = Project(),
    ) -> None:
        super().__init__()

        self._initial_noise = InitialNoiseEditor(value.initial_noise)
        self._pipeline = PipelineEditor(value.pipeline)

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._initial_noise
        yield self._pipeline

    def read(self, data: ReadData) -> Project:
        return Project(
            initial_noise = data[self._initial_noise],
            pipeline = data[self._pipeline],
        )

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {
            self._initial_noise: {},
            self._pipeline: {},
        }

        if isinstance(value := data.get("value", None), Project):
            result[self._initial_noise]["value"] = value.initial_noise
            result[self._pipeline]["value"] = value.pipeline

        if isinstance(preview_states := data.get("preview_states", None), dict):
            result[self._pipeline]["preview_states"] = preview_states

        return result
