import skimage

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage
from temporal.utils.numpy import saturate_array


class BlurringFilter(ImageFilter):
    name = "Blurring"

    radius: float = Field(0.0, name = "Radius", minimum = 0.0, maximum = 50.0, step = 0.1, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        return saturate_array(skimage.filters.gaussian(image, round(self.radius), channel_axis = -1))
