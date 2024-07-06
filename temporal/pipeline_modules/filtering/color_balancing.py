from temporal.meta.configurable import FloatParam
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.project import Project
from temporal.utils.image import NumpyImage, split_hsv, join_hsv_to_rgb
from temporal.utils.math import remap_range


class ColorBalancingFilter(ImageFilter):
    name = "Color balancing"

    brightness: float = FloatParam("Brightness", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, ui_type = "slider")
    contrast: float = FloatParam("Contrast", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, ui_type = "slider")
    saturation: float = FloatParam("Saturation", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, ui_type = "slider")

    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        npim = remap_range(npim, npim.min(), npim.max(), 0.0, self.brightness)

        npim = remap_range(npim, npim.min(), npim.max(), 0.5 - self.contrast / 2, 0.5 + self.contrast / 2)

        h, s, v = split_hsv(npim)
        s[:] = remap_range(s, s.min(), s.max(), s.min(), self.saturation)

        return join_hsv_to_rgb(h, s, v)
