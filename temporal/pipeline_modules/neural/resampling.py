from math import floor

from temporal.general_data import GeneralData
from temporal.object import Param, UndefinedValue
from temporal.pipeline_modules.neural import NeuralModule
from temporal.shared import shared
from temporal.utils import logging
from temporal.utils.collection import get_first_element
from temporal.utils.image import NumpyImage, ensure_image_dims


class ResamplingModule(NeuralModule):
    name = "Resampling"

    upscaler: str = Param("Upscaler", choices = list(shared.backend.list_upscalers()), value = get_first_element(shared.backend.list_upscalers(), UndefinedValue), ui_type = "menu")
    scale: float = Param("Scale", minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0, ui_type = "slider")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        scale = self.scale

        if scale < 1.0:
            scale = 1.0 / scale
            image = ensure_image_dims(image, size = (floor(general.image_size.x / scale), floor(general.image_size.y / scale)))

        if (result := shared.backend.upscale_image(image, self.upscaler, scale)) is not None:
            return ensure_image_dims(result, (general.image_size.x, general.image_size.y), 3)
        else:
            logging.warning("Couldn't resample an image for some reason")
            return image
