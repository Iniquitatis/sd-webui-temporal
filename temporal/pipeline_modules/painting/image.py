import numpy as np
import skimage

from temporal.general_data import GeneralData
from temporal.image_source import ImageSource
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.pipeline_modules.painting import PaintingModule
from temporal.utils.image import NumpyImage, ensure_image_dims


class ImagePaintingModule(PaintingModule):
    name = "Image"

    source: ImageSource = Param("Image source", channels = 4, factory = ImageSource)
    blurring: float = Param("Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0, ui_type = "slider")

    def draw(self, size: tuple[int, int], general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        if (image := self.source.get_image(general.initial_image, frame_index - 1)) is None:
            return np.zeros((size[1], size[0], 4))

        return ensure_image_dims(skimage.filters.gaussian(image, round(self.blurring), channel_axis = -1), size = size)
