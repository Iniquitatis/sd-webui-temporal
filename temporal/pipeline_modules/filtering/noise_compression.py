import skimage

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage


class NoiseCompressionFilter(ImageFilter):
    name = "Noise compression"

    constant: float = Field(0.0, name = "Constant", minimum = 0.0, maximum = 1.0, step = 1e-5, display = "slider")
    adaptive: float = Field(0.0, name = "Adaptive", minimum = 0.0, maximum = 1.0, step = 0.01, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        weight = 0.0

        if self.constant > 0.0:
            weight += self.constant

        if self.adaptive > 0.0:
            weight += skimage.restoration.estimate_sigma(image, average_sigmas = True, channel_axis = -1) * self.adaptive

        return skimage.restoration.denoise_tv_chambolle(image, weight = max(weight, 1e-5), channel_axis = -1)
