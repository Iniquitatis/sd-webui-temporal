import numpy as np

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage
from temporal.utils.math import quantize


class PosterizationFilter(ImageFilter):
    name = "Posterization"

    levels: int = Field(16, name = "Levels", minimum = 1, maximum = 256, step = 1, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        return quantize(image, 1.0 / self.levels, np.round)
