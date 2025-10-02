from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline import Pipeline
from temporal.pipeline_modules.control import ControlModule
from temporal.pipeline_state import PipelineResult
from temporal.utils.image import NumpyImage


class GroupModule(ControlModule):
    name = "Group"

    pipeline: Pipeline = Field(Pipeline, name = "Pipeline", display = "unpack")

    def forward(self, image: NumpyImage, general: GeneralData) -> PipelineResult:
        yield from self.pipeline.run(image, general)

    def finalize(self, general: GeneralData) -> None:
        self.pipeline.finalize(general)

    def interrupt(self, general: GeneralData) -> None:
        self.pipeline.interrupt(general)
