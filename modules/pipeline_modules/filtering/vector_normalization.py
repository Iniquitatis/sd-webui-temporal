import numpy as np

from modules.general_data import GeneralData
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage


class VectorNormalizationFilter(ImageFilter):
    name = "Vector normalization"

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        magnitude = np.linalg.norm(image[..., :3], axis = -1, keepdims = True)
        result = image.copy()
        result = np.where(image[..., :3] > 0.0, image[..., :3] / magnitude, image[..., :3])
        return result
