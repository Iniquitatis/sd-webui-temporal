from typing import Optional

import numpy as np

from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.temporal import TemporalModule
from modules.seed import Seed
from modules.utils.image import NumpyImage, ensure_image_dims
from modules.utils.math import clamp, lerp
from modules.utils.numpy import FloatArray, random_array


class RandomSamplingModule(TemporalModule):
    name = "Random sampling"

    chance: float = Field(1.0, name = "Chance", minimum = 0.0, maximum = 1.0, step = 0.001, display = "slider")
    opacity: float = Field(1.0, name = "Opacity", minimum = 0.0, maximum = 1.0, step = 0.001, display = "slider")
    use_global_seed: bool = Field(False, name = "Use global seed")
    seed: Seed = Field(Seed, name = "Seed", dependencies = {"use_global_seed": False})
    iteration: int = Field(0, flags = {"runtime"})
    buffer: Optional[FloatArray] = Field(None, flags = {"runtime"})

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        if self.buffer is None:
            self.buffer = ensure_image_dims(image.copy(), (general.image_size.x, general.image_size.y), 3)

        size = self.buffer.shape[:2]
        seed = (general.seed if self.use_global_seed else self.seed).fixed_value + self.iteration

        chance_mask = random_array(size, seed = seed) <= self.chance
        opacity_mask = random_array(
            size,
            low = clamp(self.opacity * 2.0 - 1.0, 0.0, 1.0),
            high = clamp(self.opacity * 2.0, 0.0, 1.0),
            seed = seed + 1,
        )

        self.buffer[:] = lerp(self.buffer, np.where(chance_mask[..., np.newaxis], image, self.buffer), opacity_mask[..., np.newaxis])

        self.iteration += 1

        return self.buffer.copy()
