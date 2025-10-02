import numpy as np

from modules.color import Color
from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage, apply_color_matrix


class ColorMatrixFilter(ImageFilter):
    name = "Color matrix"

    r: Color = Field(lambda: Color(1.0, 0.0, 0.0), name = "R", channels = 3)
    g: Color = Field(lambda: Color(0.0, 1.0, 0.0), name = "G", channels = 3)
    b: Color = Field(lambda: Color(0.0, 0.0, 1.0), name = "B", channels = 3)
    normalized: bool = Field(False, name = "Normalized")

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        matrix = np.array([
            self.r.to_numpy(3),
            self.g.to_numpy(3),
            self.b.to_numpy(3),
        ])

        if self.normalized:
            lengths = matrix.sum(axis = 1)[..., np.newaxis]
            matrix /= np.where(lengths > 0.0, lengths, 1.0)

        return apply_color_matrix(image, matrix)
