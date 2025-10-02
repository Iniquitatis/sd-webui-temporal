from time import sleep

from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.control import ControlModule
from modules.pipeline_state import PipelineResult, PipelineState
from modules.utils.image import NumpyImage


class DelayModule(ControlModule):
    name = "Delay"

    time: float = Field(1.0, name = "Time", minimum = 0.0, step = 0.1, suffix = " seconds", display = "box")

    def forward(self, image: NumpyImage, general: GeneralData) -> PipelineResult:
        sleep(self.time)
        yield PipelineState.finish(image = image, preview = self.preview)
