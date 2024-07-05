from typing import Optional

import numpy as np
from numpy.typing import NDArray

from temporal.meta.configurable import EnumParam, FloatParam
from temporal.meta.serializable import SerializableField as Field
from temporal.pipeline_modules.temporal import TemporalModule
from temporal.project import Project
from temporal.utils.image import NumpyImage, ensure_image_dims, match_image
from temporal.utils.numpy import saturate_array


class LimitingModule(TemporalModule):
    id = "limiting"
    name = "Limiting"

    mode: str = EnumParam("Mode", choices = [("clamp", "Clamp"), ("compress", "Compress")], value = "clamp", ui_type = "menu")
    max_difference: float = FloatParam("Maximum difference", minimum = 0.001, maximum = 1.0, step = 0.001, value = 1.0, ui_type = "slider")

    buffer: Optional[NDArray[np.float_]] = Field(None)

    def forward(self, images: list[NumpyImage], project: Project, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if self.buffer is None:
            self.buffer = np.stack([
                ensure_image_dims(image, "RGB", (project.processing.width, project.processing.height))
                for image in images
            ], 0)

        for sub, image in zip(self.buffer, images):
            a = sub
            b = match_image(image, sub)
            diff = b - a

            if self.mode == "clamp":
                np.clip(diff, -self.max_difference, self.max_difference, out = diff)
            elif self.mode == "compress":
                diff_range = np.abs(diff.max() - diff.min())
                max_diff_range = self.max_difference * 2.0

                if diff_range > max_diff_range:
                    diff *= max_diff_range / diff_range

            sub[:] = saturate_array(a + diff)

        return [sub for sub in self.buffer]

    def reset(self) -> None:
        self.buffer = None
