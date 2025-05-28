import skimage

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage


class DenoisingFilter(ImageFilter):
    name = "Denoising"

    value: float = Field(1e-5, name = "Value", minimum = 1e-5, maximum = 1.0, step = 1e-5, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        return skimage.restoration.denoise_tv_chambolle(image, weight = self.value, channel_axis = -1)
