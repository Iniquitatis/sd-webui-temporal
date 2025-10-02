import numpy as np

from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage
from modules.utils.math import quantize


class PosterizationFilter(ImageFilter):
    name = "Posterization"

    levels: int = Field(16, name = "Levels", minimum = 1, maximum = 256, step = 1, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        return quantize(image, 1.0 / self.levels, np.round)
