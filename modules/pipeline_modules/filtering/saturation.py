import numpy as np

from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage
from modules.utils.math import lerp
from modules.utils.numpy import saturate_array


class SaturationFilter(ImageFilter):
    name = "Saturation"

    mode: str = Field("bt709", name = "Mode", choices = {
        "average": "Average",
        "bt601": "BT601",
        "bt709": "BT709",
        "bt2020": "BT2020",
    }, display = "radio")
    value: float = Field(1.0, name = "Value", minimum = 0.0, maximum = 2.0, step = 0.01, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        if self.mode == "average":
            vector = 0.5, 0.5, 0.5
        elif self.mode == "bt601":
            vector = 0.299, 0.587, 0.114
        elif self.mode == "bt709":
            vector = 0.2126, 0.7152, 0.0722
        elif self.mode == "bt2020":
            vector = 0.2627, 0.678, 0.0593
        else:
            raise ValueError(self.mode)

        grayscale = np.dot(image[..., :3], vector)

        result = image.copy()
        result[..., :3] = saturate_array(lerp(grayscale.reshape(grayscale.shape + (1,)), image[..., :3], self.value))
        return result
