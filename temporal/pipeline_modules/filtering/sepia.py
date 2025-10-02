import numpy as np

from modules.general_data import GeneralData
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage, apply_color_matrix


class SepiaFilter(ImageFilter):
    name = "Sepia"

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        return apply_color_matrix(image, np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ]))
