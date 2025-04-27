from typing import Optional

import numpy as np

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.temporal import TemporalModule
from temporal.utils.image import NumpyImage, ensure_image_dims
from temporal.utils.math import clamp, lerp
from temporal.utils.numpy import FloatArray, random_array


class RandomSamplingModule(TemporalModule):
    name = "Random sampling"

    chance: float = Field(1.0, name = "Chance", minimum = 0.0, maximum = 1.0, step = 0.001, display = "slider")
    opacity: float = Field(1.0, name = "Opacity", minimum = 0.0, maximum = 1.0, step = 0.001, display = "slider")
    buffer: Optional[FloatArray] = Field(None, flags = {"private"})

    def forward(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> Optional[NumpyImage]:
        if self.buffer is None:
            self.buffer = ensure_image_dims(image.copy(), (general.image_size.x, general.image_size.y), 3)

        size = self.buffer.shape[:2]

        chance_mask = random_array(size, seed = seed) <= self.chance
        opacity_mask = random_array(
            size,
            low = clamp(self.opacity * 2.0 - 1.0, 0.0, 1.0),
            high = clamp(self.opacity * 2.0, 0.0, 1.0),
            seed = seed + 1,
        )

        self.buffer[:] = lerp(self.buffer, np.where(chance_mask[..., np.newaxis], image, self.buffer), opacity_mask[..., np.newaxis])

        return self.buffer.copy()

    def reset(self, general: GeneralData) -> None:
        self.buffer = None
