from math import floor
from typing import Optional

from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.pipeline_modules.neural import NeuralModule
from temporal.shared import shared
from temporal.utils.collection import get_first_element
from temporal.utils.image import NumpyImage, ensure_image_dims


class ResamplingModule(NeuralModule):
    name = "Resampling"

    upscaler: str = Param("Upscaler", choices = shared.backend.list_upscalers(), value = get_first_element(shared.backend.list_upscalers()), ui_type = "menu")
    scale: float = Param("Scale", minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0, ui_type = "slider")

    def forward(self, image: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> Optional[NumpyImage]:
        scale = self.scale

        if scale < 1.0:
            scale = 1.0 / scale
            image = ensure_image_dims(image, size = (floor(general.image_size.x / scale), floor(general.image_size.y / scale)))

        if (result := shared.backend.upscale_image(image, self.upscaler, scale)) is not None:
            return self._blend(image, ensure_image_dims(result, (general.image_size.x, general.image_size.y), 3))
