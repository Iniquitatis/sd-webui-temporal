import skimage

from temporal.general_data import GeneralData
from temporal.object import Param
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage


class SharpeningFilter(ImageFilter):
    name = "Sharpening"

    strength: float = Param("Strength", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, ui_type = "slider")
    radius: float = Param("Radius", minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0, ui_type = "slider")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        # NOTE: `ndim - 1` is intentional, as there's probably a bug in skimage
        return skimage.filters.unsharp_mask(image, self.radius, self.strength, channel_axis = image.ndim - 1)
