import numpy as np

from modules.color import Color
from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.painting import PaintingModule
from modules.utils.image import NumpyImage
from modules.utils.numpy import FloatArray


class PatternPaintingModule(PaintingModule):
    name = "Pattern"

    type: str = Field("horizontal_lines", name = "Type", choices = {
        "horizontal_lines": "Horizontal lines",
        "vertical_lines": "Vertical lines",
        "diagonal_lines_nw": "Diagonal lines NW",
        "diagonal_lines_ne": "Diagonal lines NE",
        "checkerboard": "Checkerboard",
    }, display = "radio");
    size: int = Field(8, name = "Size", minimum = 1, step = 1, suffix = " px", display = "box")
    color_a: Color = Field(lambda: Color(1.0, 1.0, 1.0), name = "Color A", channels = 4)
    color_b: Color = Field(lambda: Color(0.0, 0.0, 0.0), name = "Color B", channels = 4)

    def draw(self, size: tuple[int, int], general: GeneralData) -> NumpyImage:
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
