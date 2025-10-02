from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline import Pipeline
from temporal.pipeline_modules.control import ControlModule
from temporal.pipeline_state import PipelineResult, PipelineState
from temporal.utils.image import NumpyImage


class BranchModule(ControlModule):
    name = "Branch"

    pipeline: Pipeline = Field(Pipeline, name = "Pipeline", display = "unpack")

    def forward(self, image: NumpyImage, general: GeneralData) -> PipelineResult:
        for state in self.pipeline.run(image, general):
            match state:
                case PipelineState.progress():
                    state.preview = self.preview and state.preview
                    yield state
                case PipelineState.finish():
                    yield PipelineState.progress(image = state.image, preview = self.preview and state.preview)
                case PipelineState.fail():
                    yield state
                    return

    def finalize(self, general: GeneralData) -> None:
        self.pipeline.finalize(general)

    def interrupt(self, general: GeneralData) -> None:
        self.pipeline.interrupt(general)
