from typing import Optional

import numpy as np

from temporal.general_data import GeneralData
from temporal.object import Field, Param
from temporal.pipeline_modules.temporal import TemporalModule
from temporal.utils.image import NumpyImage, ensure_image_dims, match_image
from temporal.utils.numpy import FloatArray, average_array, make_eased_weight_array, saturate_array


class AveragingModule(TemporalModule):
    name = "Averaging"

    frames: int = Param("Frame count", minimum = 1, step = 1, value = 1, ui_type = "box")
    trimming: float = Param("Trimming", minimum = 0.0, maximum = 0.5, step = 0.01, value = 0.0, ui_type = "slider")
    easing: float = Param("Easing", minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0, ui_type = "slider")
    preference: float = Param("Preference", minimum = -2.0, maximum = 2.0, step = 0.1, value = 0.0, ui_type = "slider")

    buffer: Optional[FloatArray] = Field(None, flags = {"private"})
    last_index: int = Field(0, flags = {"private"})

    def forward(self, image: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> Optional[NumpyImage]:
        if self.buffer is None:
            self.buffer = np.repeat(
                ensure_image_dims(image, (general.image_size.x, general.image_size.y), 3)[np.newaxis, ...],
                self.frames,
                axis = 0,
            )
            self.last_index = 0

        self.buffer[self.last_index] = match_image(image, self.buffer[0])

        self.last_index += 1
        self.last_index %= self.frames

        return self.buffer[0].copy() if self.frames == 1 else saturate_array(average_array(
            self.buffer,
            axis = 0,
            trim = self.trimming,
            power = self.preference + 1.0,
            weights = np.roll(make_eased_weight_array(self.frames, self.easing), self.last_index),
        ))

    def reset(self, general: GeneralData) -> None:
        self.buffer = None
        self.last_index = 0
