from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline import Pipeline
from modules.pipeline_modules.control import ControlModule
from modules.pipeline_state import PipelineResult, PipelineState
from modules.utils.image import NumpyImage


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

    def interrupt(self, general: GeneralData) -> None:
        self.pipeline.interrupt(general)
