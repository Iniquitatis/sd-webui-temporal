import numpy as np

from modules.color import Color
from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.painting import PaintingModule
from modules.utils.image import NumpyImage


class ColorPaintingModule(PaintingModule):
    name = "Color"

    color: Color = Field(Color, name = "Color", channels = 4)

    def draw(self, size: tuple[int, int], general: GeneralData) -> NumpyImage:
        return np.full((size[1], size[0], 4), self.color.to_numpy(4))
