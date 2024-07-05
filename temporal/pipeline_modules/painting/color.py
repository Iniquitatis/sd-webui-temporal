import numpy as np

from temporal.color import Color
from temporal.meta.configurable import ColorParam
from temporal.pipeline_modules.painting import PaintingModule
from temporal.project import Project
from temporal.utils.image import NumpyImage


class ColorPaintingModule(PaintingModule):
    id = "color_painting"
    name = "Color"

    color: Color = ColorParam("Color", channels = 4)

    def draw(self, size: tuple[int, int], parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        return np.full((size[1], size[0], 4), self.color.to_numpy(4))
