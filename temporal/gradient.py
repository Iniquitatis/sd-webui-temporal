from typing import Literal

import numpy as np
from numpy.typing import NDArray

from temporal.color import Color
from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.utils.math import lerp


class Gradient(Serializable):
    type: Literal["linear", "radial"] = Field("linear")
    start_x: float = Field(0.0)
    start_y: float = Field(0.0)
    end_x: float = Field(1.0)
    end_y: float = Field(1.0)
    start_color: Color = Field(factory = lambda: Color(1.0, 1.0, 1.0))
    end_color: Color = Field(factory = lambda: Color(0.0, 0.0, 0.0))

    def generate(self, shape: tuple[int, ...], show_points: bool = False) -> NDArray[np.float_]:
        start_x, start_y = self.start_x * shape[1], self.start_y * shape[0]
        end_x, end_y = self.end_x * shape[1], self.end_y * shape[0]

        y, x = np.indices(shape[:2])

        if self.type == "linear":
            point_to_start = (
                (x - start_x) * (end_x - start_x) +
                (y - start_y) * (end_y - start_y)
            )

            end_to_start = (
                (end_x - start_x) * (end_x - start_x) +
                (end_y - start_y) * (end_y - start_y)
            )

            factor = np.clip(point_to_start / end_to_start, 0.0, 1.0)

        elif self.type == "radial":
            point_to_start = np.sqrt(
                np.power(x - start_x, 2.0) +
                np.power(y - start_y, 2.0),
            )

            end_to_start = np.sqrt(
                np.power(end_x - start_x, 2.0) +
                np.power(end_y - start_y, 2.0),
            )

            factor = np.clip(point_to_start / end_to_start, 0.0, 1.0)

        else:
            raise NotImplementedError

        result = lerp(
            self.start_color.to_numpy(shape[-1]),
            self.end_color.to_numpy(shape[-1]),
            factor[..., np.newaxis],
        )

        if show_points:
            start_x, start_y = int(start_x), int(start_y)
            end_x, end_y = int(end_x), int(end_y)

            if start_x in range(shape[1]) and start_y in range(shape[0]):
                result[start_y, start_x] = [0.0, 1.0, 0.0, 1.0]

            if end_x in range(shape[1]) and end_y in range(shape[0]):
                result[end_y, end_x] = [1.0, 0.0, 0.0, 1.0]

        return result
