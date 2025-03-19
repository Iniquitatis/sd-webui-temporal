from temporal.general_data import GeneralData
from temporal.object import Param
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage, join_hsv_to_rgb, split_hsv
from temporal.utils.math import remap_range
from temporal.utils.numpy import saturate_array


class ColorBalancingFilter(ImageFilter):
    name = "Color balancing"

    brightness: float = Param("Brightness", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, ui_type = "slider")
    contrast: float = Param("Contrast", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, ui_type = "slider")
    saturation: float = Param("Saturation", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, ui_type = "slider")

    # TODO: Make non-idempotent
    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        image = remap_range(image, image.min(), image.max(), 0.0, self.brightness)

        image = remap_range(image, image.min(), image.max(), 0.5 - self.contrast / 2, 0.5 + self.contrast / 2)

        h, s, v = split_hsv(image)
        s[:] = remap_range(s, s.min(), s.max(), s.min(), self.saturation)

        return saturate_array(join_hsv_to_rgb(h, s, v))
