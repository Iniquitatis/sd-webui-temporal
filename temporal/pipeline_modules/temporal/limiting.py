from typing import Optional

import numpy as np

from temporal.general_data import GeneralData
from temporal.object import Field, Param
from temporal.pipeline_modules.temporal import TemporalModule
from temporal.utils.image import NumpyImage, ensure_image_dims, match_image
from temporal.utils.numpy import FloatArray, saturate_array


class LimitingModule(TemporalModule):
    name = "Limiting"

    mode: str = Param("Mode", choices = {"clamp": "Clamp", "compress": "Compress"}, value = "clamp", ui_type = "radio")
    max_difference: float = Param("Maximum difference", minimum = 0.001, maximum = 1.0, step = 0.001, value = 1.0, ui_type = "slider")

    buffer: Optional[FloatArray] = Field(None, flags = {"private"})

    def forward(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> Optional[NumpyImage]:
        if self.buffer is None:
            self.buffer = ensure_image_dims(image.copy(), (general.image_size.x, general.image_size.y), 3)

        a = self.buffer
        b = match_image(image, self.buffer)
        diff = b - a

        if self.mode == "clamp":
            np.clip(diff, -self.max_difference, self.max_difference, out = diff)
        elif self.mode == "compress":
            diff_range = np.abs(diff.max() - diff.min())
            max_diff_range = self.max_difference * 2.0

            if diff_range > max_diff_range:
                diff *= max_diff_range / diff_range

        self.buffer[:] = saturate_array(a + diff)

        return self.buffer.copy()

    def reset(self, general: GeneralData) -> None:
        self.buffer = None
