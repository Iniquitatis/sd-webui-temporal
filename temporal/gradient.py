from typing import Literal

import numpy as np
from numpy.typing import NDArray

from temporal.color import Color
from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.utils.math import lerp
from temporal.vector import FloatVector


class Gradient(Serializable):
    type: Literal["linear", "radial"] = Field("linear")
    start: FloatVector = Field(factory = lambda: FloatVector(0.0, 0.0))
    end: FloatVector = Field(factory = lambda: FloatVector(1.0, 1.0))
    start_color: Color = Field(factory = lambda: Color(1.0, 1.0, 1.0))
    end_color: Color = Field(factory = lambda: Color(0.0, 0.0, 0.0))

    def generate(self, shape: tuple[int, ...], show_points: bool = False) -> NDArray[np.float64]:
        start = self.start.to_numpy()[[1, 0]] * shape[:2]
        end = self.end.to_numpy()[[1, 0]] * shape[:2]

        coords = np.indices(shape[:2]).transpose(1, 2, 0)

        if self.type == "linear":
            point_to_start = np.dot(coords - start, end - start)
            end_to_start = np.dot(end - start, end - start)
            factor = np.clip(point_to_start / end_to_start, 0.0, 1.0)

        elif self.type == "radial":
            point_to_start = np.sqrt(np.power(coords - start, 2.0).sum(-1))
            end_to_start = np.sqrt(np.power(end - start, 2.0).sum(-1))
            factor = np.clip(point_to_start / end_to_start, 0.0, 1.0)

        else:
            raise NotImplementedError

        result = lerp(
            self.start_color.to_numpy(shape[-1]),
            self.end_color.to_numpy(shape[-1]),
            factor[..., np.newaxis],
        )

        if show_points:
            start = start.astype(np.int32)
            end = end.astype(np.int32)

            if all((start >= 0) & (start < shape[:2])):
                result[*start] = [0.0, 1.0, 0.0, 1.0][:shape[-1]]

            if all((end >= 0) & (end < shape[:2])):
                result[*end] = [1.0, 0.0, 0.0, 1.0][:shape[-1]]

        return result
