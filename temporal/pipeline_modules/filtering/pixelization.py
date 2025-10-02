import numpy as np

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage


class PixelizationFilter(ImageFilter):
    name = "Pixelization"

    pixel_size: int = Field(1, name = "Pixel size", minimum = 1, step = 1, suffix = " px", display = "box")

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        height, width = image.shape[:2]

        y, x = np.indices((height, width))

        return np.mean([
            image[
                np.clip(y // self.pixel_size * self.pixel_size + j, 0, height - 1),
                np.clip(x // self.pixel_size * self.pixel_size + i, 0, width  - 1),
            ]
            for j in range(self.pixel_size)
            for i in range(self.pixel_size)
        ], axis = 0)
