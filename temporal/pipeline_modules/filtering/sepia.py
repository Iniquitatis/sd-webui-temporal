import numpy as np

from temporal.general_data import GeneralData
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage, apply_color_matrix


class SepiaFilter(ImageFilter):
    name = "Sepia"

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        return apply_color_matrix(image, np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ]))
