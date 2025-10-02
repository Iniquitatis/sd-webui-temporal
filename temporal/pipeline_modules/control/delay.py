from time import sleep

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.control import ControlModule
from temporal.pipeline_state import PipelineResult, PipelineState
from temporal.utils.image import NumpyImage


class DelayModule(ControlModule):
    name = "Delay"

    time: float = Field(1.0, name = "Time", minimum = 0.0, step = 0.1, suffix = " seconds", display = "box")

    def forward(self, image: NumpyImage, general: GeneralData) -> PipelineResult:
        sleep(self.time)
        yield PipelineState.finish(image = image, preview = self.preview)
