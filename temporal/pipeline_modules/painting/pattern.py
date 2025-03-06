import numpy as np

from temporal.color import Color
from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.pipeline_modules.painting import PaintingModule
from temporal.utils.image import NumpyImage
from temporal.utils.numpy import FloatArray


class PatternPaintingModule(PaintingModule):
    name = "Pattern"

    type: str = Param("Type", choices = {
        "horizontal_lines": "Horizontal lines",
        "vertical_lines": "Vertical lines",
        "diagonal_lines_nw": "Diagonal lines NW",
        "diagonal_lines_ne": "Diagonal lines NE",
        "checkerboard": "Checkerboard",
    }, value = "horizontal_lines", ui_type = "radio");
    size: int = Param("Size", minimum = 1, step = 1, value = 8, ui_type = "box")
    color_a: Color = Param("Color A", channels = 4, value = lambda: Color(1.0, 1.0, 1.0))
    color_b: Color = Param("Color B", channels = 4, value = lambda: Color(0.0, 0.0, 0.0))

    def draw(self, size: tuple[int, int], general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        return self._generate((size[1], size[0], 4))

    def _generate(self, shape: tuple[int, ...]) -> FloatArray:
        y, x = np.indices(shape[:2])

        if self.type == "horizontal_lines":
            pattern = (y // self.size % 2 == 0)

        elif self.type == "vertical_lines":
            pattern = (x // self.size % 2 == 0)

        elif self.type == "diagonal_lines_nw":
            pattern = ((x - y) // self.size % 2 == 0)

        elif self.type == "diagonal_lines_ne":
            pattern = ((x + y) // self.size % 2 == 0)

        elif self.type == "checkerboard":
            pattern = (x // self.size + y // self.size) % 2 == 0

        else:
            raise NotImplementedError

        return np.where(pattern[..., np.newaxis], self.color_a.to_numpy(shape[-1]), self.color_b.to_numpy(shape[-1]))
