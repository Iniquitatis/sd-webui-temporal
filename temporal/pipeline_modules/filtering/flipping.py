import numpy as np

from temporal.general_data import GeneralData
from temporal.object import Param
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage


class FlippingFilter(ImageFilter):
    name = "Flipping"

    horizontal: bool = Param("Horizontal", value = False)
    vertical: bool = Param("Vertical", value = False)

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        image = image.copy()

        if self.horizontal:
            image = np.flip(image, axis = 1)

        if self.vertical:
            image = np.flip(image, axis = 0)

        return image
