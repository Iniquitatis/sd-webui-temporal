from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline import Pipeline
from modules.pipeline_modules.control import ControlModule
from modules.pipeline_state import PipelineResult
from modules.utils.image import NumpyImage


class GroupModule(ControlModule):
    name = "Group"

    pipeline: Pipeline = Field(Pipeline, name = "Pipeline", display = "unpack")

    def forward(self, image: NumpyImage, general: GeneralData) -> PipelineResult:
        yield from self.pipeline.run(image, general)

    def finalize(self, general: GeneralData) -> None:
        self.pipeline.finalize(general)

    def interrupt(self, general: GeneralData) -> None:
        self.pipeline.interrupt(general)
