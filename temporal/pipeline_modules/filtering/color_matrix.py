import numpy as np

from temporal.color import Color
from temporal.general_data import GeneralData
from temporal.object import Param
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage, apply_color_matrix


class ColorMatrixFilter(ImageFilter):
    name = "Color matrix"

    r: Color = Param("R", channels = 3, value = lambda: Color(1.0, 0.0, 0.0))
    g: Color = Param("G", channels = 3, value = lambda: Color(0.0, 1.0, 0.0))
    b: Color = Param("B", channels = 3, value = lambda: Color(0.0, 0.0, 1.0))
    normalized: bool = Param("Normalized", value = False)

    def process(self, image: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        matrix = np.array([
            self.r.to_numpy(3),
            self.g.to_numpy(3),
            self.b.to_numpy(3),
        ])

        if self.normalized:
            lengths = matrix.sum(axis = 1)[..., np.newaxis]
            matrix /= np.where(lengths > 0.0, lengths, 1.0)

        return apply_color_matrix(image, matrix)
