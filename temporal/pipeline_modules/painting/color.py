import numpy as np

from temporal.color import Color
from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.pipeline_modules.painting import PaintingModule
from temporal.utils.image import NumpyImage


class ColorPaintingModule(PaintingModule):
    name = "Color"

    color: Color = Param("Color", channels = 4, value = Color)

    def draw(self, size: tuple[int, int], general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        return np.full((size[1], size[0], 4), self.color.to_numpy(4))
