from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage
from temporal.utils.numpy import saturate_array


class BrightnessFilter(ImageFilter):
    name = "Brightness"

    value: float = Field(1.0, name = "Value", minimum = 0.0, maximum = 2.0, step = 0.01, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        result = image.copy()
        result[..., :3] = saturate_array(image[..., :3] * self.value)
        return result
