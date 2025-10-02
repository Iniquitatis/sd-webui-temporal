import skimage

from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage
from modules.utils.numpy import saturate_array


class BlurringFilter(ImageFilter):
    name = "Blurring"

    radius: float = Field(0.0, name = "Radius", minimum = 0.0, maximum = 50.0, step = 0.1, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        return saturate_array(skimage.filters.gaussian(image, round(self.radius), channel_axis = -1))
