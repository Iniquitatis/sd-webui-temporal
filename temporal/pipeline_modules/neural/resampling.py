from math import floor

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.neural import NeuralModule
from temporal.pipeline_state import PipelineResult, PipelineState
from temporal.shared import shared
from temporal.utils.collection import get_first_element
from temporal.utils.image import NumpyImage, ensure_image_dims


class ResamplingModule(NeuralModule):
    name = "Resampling"

    upscaler: str = Field(lambda: get_first_element(shared.backend.list_upscalers(), ("", ""))[0], name = "Upscaler", choices = lambda: dict(shared.backend.list_upscalers()), display = "menu")
    scale: float = Field(1.0, name = "Scale", minimum = 0.25, maximum = 4.0, step = 0.25, display = "slider")

    def forward(self, image: NumpyImage, general: GeneralData) -> PipelineResult:
        scale = self.scale

        if scale < 1.0:
            scale = 1.0 / scale
            image = ensure_image_dims(image, size = (floor(general.image_size.x / scale), floor(general.image_size.y / scale)))

        if (result := shared.backend.upscale_image(image, self.upscaler, scale)) is not None:
            yield PipelineState.finish(image = self._blend(image, ensure_image_dims(result, (general.image_size.x, general.image_size.y), 3)), preview = self.preview)
        else:
            yield PipelineState.fail()

    def interrupt(self, general: GeneralData) -> None:
        shared.backend.interrupt()
