import numpy as np

from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage


class FlippingFilter(ImageFilter):
    name = "Flipping"

    horizontal: bool = Field(False, name = "Horizontal")
    vertical: bool = Field(False, name = "Vertical")

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        image = image.copy()

        if self.horizontal:
            image = np.flip(image, axis = 1)

        if self.vertical:
            image = np.flip(image, axis = 0)

        return image
