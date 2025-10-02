import numpy as np

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage


class GammaFilter(ImageFilter):
    name = "Gamma"

    input: float = Field(1.0, name = "Input", minimum = 0.01, maximum = 4.0, step = 0.01, display = "slider")
    output: float = Field(1.0, name = "Output", minimum = 0.01, maximum = 4.0, step = 0.01, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        return np.power(image, self.input / self.output)
