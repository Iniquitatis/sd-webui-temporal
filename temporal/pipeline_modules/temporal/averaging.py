from typing import Optional

import numpy as np

from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.temporal import TemporalModule
from modules.utils.image import NumpyImage, ensure_image_dims, match_image
from modules.utils.numpy import FloatArray, average_array, make_eased_weight_array, saturate_array


class AveragingModule(TemporalModule):
    name = "Averaging"

    sample_count: int = Field(1, name = "Sample count", minimum = 1, step = 1, display = "box")
    trimming: float = Field(0.0, name = "Trimming", minimum = 0.0, maximum = 0.5, step = 0.01, display = "slider")
    easing: float = Field(0.0, name = "Easing", minimum = 0.0, maximum = 16.0, step = 0.1, display = "slider")
    preference: float = Field(0.0, name = "Preference", minimum = -2.0, maximum = 2.0, step = 0.1, display = "slider")
    buffer: Optional[FloatArray] = Field(None, flags = {"runtime"})
    last_index: int = Field(0, flags = {"runtime"})

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        if self.buffer is None:
            self.buffer = np.repeat(
                ensure_image_dims(image, (general.image_size.x, general.image_size.y), 3)[np.newaxis, ...],
                self.sample_count,
                axis = 0,
            )
            self.last_index = 0

        self.buffer[self.last_index] = match_image(image, self.buffer[0])

        self.last_index += 1
        self.last_index %= self.sample_count

        return self.buffer[0].copy() if self.sample_count == 1 else saturate_array(average_array(
            self.buffer,
            axis = 0,
            trim = self.trimming,
            power = self.preference + 1.0,
            weights = np.roll(make_eased_weight_array(self.sample_count, self.easing), self.last_index),
        ))
