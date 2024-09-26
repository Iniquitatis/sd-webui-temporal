import numpy as np

from temporal.meta.configurable import FloatParam
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.project import Project
from temporal.utils.image import NumpyImage
from temporal.utils.math import lerp


class TilingFilter(ImageFilter):
    name = "Tiling"

    exponent: float = FloatParam("Exponent", minimum = 1.0, maximum = 16.0, step = 0.1, value = 1.0, ui_type = "slider")

    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        h, w, _ = npim.shape

        hh, hw = h // 2, w // 2

        l_mask = np.repeat(np.power(np.linspace(0.0, 1.0, hw), self.exponent).reshape((1, hw, 1)), h, axis = 0)
        r_mask = np.repeat(np.power(np.linspace(1.0, 0.0, hw), self.exponent).reshape((1, hw, 1)), h, axis = 0)
        t_mask = np.repeat(np.power(np.linspace(0.0, 1.0, hh), self.exponent).reshape((hh, 1, 1)), w, axis = 1)
        b_mask = np.repeat(np.power(np.linspace(1.0, 0.0, hh), self.exponent).reshape((hh, 1, 1)), w, axis = 1)

        x_tiled = npim.copy()
        x_tiled[:, hw:] = lerp(npim[:, hw:], npim[:, :hw], l_mask)
        x_tiled[:, :hw] = lerp(npim[:, :hw], npim[:, hw:], r_mask)

        y_tiled = x_tiled.copy()
        y_tiled[hh:, :] = lerp(x_tiled[hh:, :], x_tiled[:hh, :], t_mask)
        y_tiled[:hh, :] = lerp(x_tiled[:hh, :], x_tiled[hh:, :], b_mask)

        return y_tiled
