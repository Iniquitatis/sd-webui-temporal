import numpy as np

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage
from temporal.utils.math import lerp
from temporal.utils.numpy import saturate_array


class ContrastFilter(ImageFilter):
    name = "Contrast"

    value: float = Field(1.0, name = "Value", minimum = 0.0, maximum = 2.0, step = 0.01, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        result = image.copy()
        result[..., :3] = saturate_array(lerp(np.full_like(image[..., :3], 0.5), image[..., :3], self.value))
        return result
