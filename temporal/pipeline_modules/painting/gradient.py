import numpy as np

from temporal.color import Color
from temporal.general_data import GeneralData
from temporal.object import Param
from temporal.pipeline_modules.painting import PaintingModule
from temporal.utils.image import NumpyImage
from temporal.utils.math import lerp
from temporal.utils.numpy import FloatArray, IntType
from temporal.vector import FloatVector


class GradientPaintingModule(PaintingModule):
    name = "Gradient"

    type: str = Param("Type", choices = {"linear": "Linear", "radial": "Radial"}, value = "linear", ui_type = "radio")
    start: FloatVector = Param("Start", axes = ["X", "Y"], step = 0.01, value = lambda: FloatVector(0.0, 0.0), ui_type = "box")
    end: FloatVector = Param("End", axes = ["X", "Y"], step = 0.01, value = lambda: FloatVector(1.0, 1.0), ui_type = "box")
    start_color: Color = Param("Start color", channels = 4, value = lambda: Color(0.0, 0.0, 0.0))
    end_color: Color = Param("End color", channels = 4, value = lambda: Color(1.0, 1.0, 1.0))

    def draw(self, size: tuple[int, int], general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        return self._generate((size[1], size[0], 4))

    def _generate(self, shape: tuple[int, ...], show_points: bool = False) -> FloatArray:
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

        # TODO: Extract into some sort of "show_gizmos" method. Or better yet,
        # "gizmo_data" and let the frontend do the dirty work.
        if show_points:
            start = start.astype(IntType)
            end = end.astype(IntType)

            if all((start >= 0) & (start < shape[:2])):
                result[*start] = [0.0, 1.0, 0.0, 1.0][:shape[-1]]

            if all((end >= 0) & (end < shape[:2])):
                result[*end] = [1.0, 0.0, 0.0, 1.0][:shape[-1]]

        return result
