import numpy as np

from temporal.meta.configurable import IntParam
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.project import Project
from temporal.utils.image import NumpyImage


class PixelizationFilter(ImageFilter):
    id = "pixelization"
    name = "Pixelization"

    pixel_size: int = IntParam("Pixel size", minimum = 1, step = 1, value = 1, ui_type = "box")

    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        height, width = npim.shape[:2]

        y, x = np.indices((height, width))

        return np.mean([
            npim[
                np.clip(y // self.pixel_size * self.pixel_size + j, 0, height - 1),
                np.clip(x // self.pixel_size * self.pixel_size + i, 0, width  - 1),
            ]
            for j in range(self.pixel_size)
            for i in range(self.pixel_size)
        ], axis = 0)
