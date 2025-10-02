import numpy as np

from temporal.general_data import GeneralData
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage


class VectorNormalizationFilter(ImageFilter):
    name = "Vector normalization"

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        magnitude = np.linalg.norm(image[..., :3], axis = -1, keepdims = True)
        result = image.copy()
        result = np.where(image[..., :3] > 0.0, image[..., :3] / magnitude, image[..., :3])
        return result
