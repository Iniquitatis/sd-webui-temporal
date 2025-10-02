import numpy as np

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage
from temporal.utils.math import lerp


class TilingFilter(ImageFilter):
    name = "Tiling"

    exponent: float = Field(1.0, name = "Exponent", minimum = 1.0, maximum = 16.0, step = 0.1, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        h, w, _ = image.shape

        hh, hw = h // 2, w // 2

        l_mask = np.repeat(np.power(np.linspace(0.0, 1.0, hw), self.exponent).reshape((1, hw, 1)), h, axis = 0)
        r_mask = np.repeat(np.power(np.linspace(1.0, 0.0, hw), self.exponent).reshape((1, hw, 1)), h, axis = 0)
        t_mask = np.repeat(np.power(np.linspace(0.0, 1.0, hh), self.exponent).reshape((hh, 1, 1)), w, axis = 1)
        b_mask = np.repeat(np.power(np.linspace(1.0, 0.0, hh), self.exponent).reshape((hh, 1, 1)), w, axis = 1)

        x_tiled = image.copy()
        x_tiled[:, hw:] = lerp(image[:, hw:], image[:, :hw], l_mask)
        x_tiled[:, :hw] = lerp(image[:, :hw], image[:, hw:], r_mask)

        y_tiled = x_tiled.copy()
        y_tiled[hh:, :] = lerp(x_tiled[hh:, :], x_tiled[:hh, :], t_mask)
        y_tiled[:hh, :] = lerp(x_tiled[:hh, :], x_tiled[hh:, :], b_mask)

        return y_tiled
