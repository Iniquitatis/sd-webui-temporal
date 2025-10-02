import numpy as np

from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage
from modules.utils.math import lerp
from modules.utils.numpy import saturate_array


class ContrastFilter(ImageFilter):
    name = "Contrast"

    value: float = Field(1.0, name = "Value", minimum = 0.0, maximum = 2.0, step = 0.01, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        result = image.copy()
        result[..., :3] = saturate_array(lerp(np.full_like(image[..., :3], 0.5), image[..., :3], self.value))
        return result
