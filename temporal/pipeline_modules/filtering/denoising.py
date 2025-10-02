import skimage

from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage


class DenoisingFilter(ImageFilter):
    name = "Denoising"

    value: float = Field(1e-5, name = "Value", minimum = 1e-5, maximum = 1.0, step = 1e-5, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        return skimage.restoration.denoise_tv_chambolle(image, weight = self.value, channel_axis = -1)
