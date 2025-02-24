from math import floor
from typing import Optional

from temporal.general_data import GeneralData
from temporal.meta.configurable import EnumParam, FloatParam
from temporal.pipeline_modules.neural import NeuralModule
from temporal.shared import shared
from temporal.utils.collection import get_first_element
from temporal.utils.image import NumpyImage, ensure_image_dims


class ResamplingModule(NeuralModule):
    name = "Resampling"

    upscaler: str = EnumParam("Upscaler", choices = shared.backend.list_upscalers(), value = get_first_element(shared.backend.list_upscalers()), ui_type = "menu")
    scale: float = FloatParam("Scale", minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0, ui_type = "slider")

    def forward(self, images: list[NumpyImage], general: GeneralData, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        def resample(im: NumpyImage) -> NumpyImage:
            scale = self.scale

            if scale < 1.0:
                scale = 1.0 / scale
                im = ensure_image_dims(im, size = (floor(general.image_size.x / scale), floor(general.image_size.y / scale)))

            if (result := shared.backend.upscale_image(im, self.upscaler, scale)) is not None:
                return result
            else:
                raise Exception

        return [
            self._blend(im, ensure_image_dims(resample(im), (general.image_size.x, general.image_size.y), 3))
            for im in images
        ]
