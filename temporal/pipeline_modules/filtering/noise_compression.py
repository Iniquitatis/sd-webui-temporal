import skimage

from temporal.meta.configurable import FloatParam
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.project import Project
from temporal.utils.image import NumpyImage


class NoiseCompressionFilter(ImageFilter):
    name = "Noise compression"

    constant: float = FloatParam("Constant", minimum = 0.0, maximum = 1.0, step = 1e-5, value = 0.0, ui_type = "slider")
    adaptive: float = FloatParam("Adaptive", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, ui_type = "slider")

    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        weight = 0.0

        if self.constant > 0.0:
            weight += self.constant

        if self.adaptive > 0.0:
            weight += skimage.restoration.estimate_sigma(npim, average_sigmas = True, channel_axis = -1) * self.adaptive

        return skimage.restoration.denoise_tv_chambolle(npim, weight = max(weight, 1e-5), channel_axis = -1)
