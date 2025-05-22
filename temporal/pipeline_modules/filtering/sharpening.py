import skimage

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage


class SharpeningFilter(ImageFilter):
    name = "Sharpening"

    strength: float = Field(0.0, name = "Strength", minimum = 0.0, maximum = 2.0, step = 0.01, display = "slider")
    radius: float = Field(0.0, name = "Radius", minimum = 0.0, maximum = 5.0, step = 0.1, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        # NOTE: `ndim - 1` is intentional, as there's probably a bug in skimage
        return skimage.filters.unsharp_mask(image, self.radius, self.strength, channel_axis = image.ndim - 1)
