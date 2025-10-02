import numpy as np

from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage


class SymmetryFilter(ImageFilter):
    name = "Symmetry"

    horizontal: bool = Field(False, name = "Horizontal")
    vertical: bool = Field(False, name = "Vertical")

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        height, width = image.shape[:2]
        image = image.copy()

        if self.horizontal:
            image[:, width // 2:] = np.flip(image[:, :width // 2], axis = 1)

        if self.vertical:
            image[height // 2:, :] = np.flip(image[:height // 2, :], axis = 0)

        return image
