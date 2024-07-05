from typing import Literal

import numpy as np
from numpy.typing import NDArray

from temporal.color import Color
from temporal.meta.serializable import Serializable, SerializableField as Field


class Pattern(Serializable):
    type: Literal["horizontal_lines", "vertical_lines", "diagonal_lines_nw", "diagonal_lines_ne", "checkerboard"] = Field("horizontal_lines")
    size: int = Field(8)
    color_a: Color = Field(factory = lambda: Color(1.0, 1.0, 1.0))
    color_b: Color = Field(factory = lambda: Color(0.0, 0.0, 0.0))

    def generate(self, shape: tuple[int, ...]) -> NDArray[np.float_]:
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
