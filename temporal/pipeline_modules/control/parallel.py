import numpy as np

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline import Pipeline
from temporal.pipeline_modules.control import ControlModule
from temporal.pipeline_state import PipelineResult, PipelineState
from temporal.utils.image import NumpyImage
from temporal.utils.numpy import average_array, make_eased_weight_array, saturate_array


class ParallelModule(ControlModule):
    name = "Parallel"

    count: int = Field(1, name = "Count", minimum = 1, step = 1, display = "box")
    trimming: float = Field(0.0, name = "Trimming", minimum = 0.0, maximum = 0.5, step = 0.01, display = "slider")
    easing: float = Field(0.0, name = "Easing", minimum = 0.0, maximum = 16.0, step = 0.1, display = "slider")
    preference: float = Field(0.0, name = "Preference", minimum = -2.0, maximum = 2.0, step = 0.1, display = "slider")
    pipeline: Pipeline = Field(Pipeline, name = "Pipeline", display = "unpack")

    def forward(self, image: NumpyImage, general: GeneralData) -> PipelineResult:
        images: list[NumpyImage] = []

        for _ in range(self.count):
            for state in self.pipeline.run(image, general):
                match state:
                    case PipelineState.progress():
                        state.preview = self.preview and state.preview
                        yield state
                    case PipelineState.finish():
                        images.append(state.image)
                        yield PipelineState.progress(image = state.image, preview = self.preview and state.preview)
                    case PipelineState.fail():
                        yield state
                        return

        yield PipelineState.finish(image = images[0] if self.count == 1 else saturate_array(average_array(
            np.array(images),
            axis = 0,
            trim = self.trimming,
            power = self.preference + 1.0,
            weights = make_eased_weight_array(self.count, self.easing),
        )), preview = self.preview)

    def finalize(self, general: GeneralData) -> None:
        self.pipeline.finalize(general)

    def interrupt(self, general: GeneralData) -> None:
        self.pipeline.interrupt(general)
