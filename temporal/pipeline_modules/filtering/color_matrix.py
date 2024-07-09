import numpy as np

from temporal.color import Color
from temporal.meta.configurable import BoolParam, ColorParam
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.project import Project
from temporal.utils.image import NumpyImage, apply_color_matrix


class ColorMatrixFilter(ImageFilter):
    name = "Color matrix"

    r: Color = ColorParam("R", channels = 3, factory = lambda: Color(1.0, 0.0, 0.0))
    g: Color = ColorParam("G", channels = 3, factory = lambda: Color(0.0, 1.0, 0.0))
    b: Color = ColorParam("B", channels = 3, factory = lambda: Color(0.0, 0.0, 1.0))
    normalized: bool = BoolParam("Normalized", value = False)

    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        matrix = np.array([
            self.r.to_numpy(3),
            self.g.to_numpy(3),
            self.b.to_numpy(3),
        ])

        if self.normalized:
            lengths = matrix.sum(axis = 1)[..., np.newaxis]
            matrix /= np.where(lengths > 0.0, lengths, 1.0)

        return apply_color_matrix(npim, matrix)
