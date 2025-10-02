from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline import Pipeline
from temporal.pipeline_modules.control import ControlModule
from temporal.pipeline_state import PipelineResult, PipelineState
from temporal.utils.image import NumpyImage


class RecursionModule(ControlModule):
    name = "Recursion"

    count: int = Field(1, name = "Count", minimum = 1, step = 1, display = "box")
    pipeline: Pipeline = Field(Pipeline, name = "Pipeline", display = "unpack")

    def forward(self, image: NumpyImage, general: GeneralData) -> PipelineResult:
        result = PipelineState.finish(image = image, preview = self.preview)

        for _ in range(self.count):
            for state in self.pipeline.run(result.image, general):
                match state:
                    case PipelineState.progress():
                        state.preview = self.preview and state.preview
                        yield state
                    case PipelineState.finish():
                        state.preview = self.preview and state.preview
                        result = state
                        yield PipelineState.progress(image = state.image, preview = state.preview)
                    case PipelineState.fail():
                        yield state
                        return

        yield result

    def finalize(self, general: GeneralData) -> None:
        self.pipeline.finalize(general)

    def interrupt(self, general: GeneralData) -> None:
        self.pipeline.interrupt(general)
