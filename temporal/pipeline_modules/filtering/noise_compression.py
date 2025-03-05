import skimage

from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage


class NoiseCompressionFilter(ImageFilter):
    name = "Noise compression"

    constant: float = Param("Constant", minimum = 0.0, maximum = 1.0, step = 1e-5, value = 0.0, ui_type = "slider")
    adaptive: float = Param("Adaptive", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, ui_type = "slider")

    def process(self, image: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        weight = 0.0

        if self.constant > 0.0:
            weight += self.constant

        if self.adaptive > 0.0:
            weight += skimage.restoration.estimate_sigma(image, average_sigmas = True, channel_axis = -1) * self.adaptive

        return skimage.restoration.denoise_tv_chambolle(image, weight = max(weight, 1e-5), channel_axis = -1)
