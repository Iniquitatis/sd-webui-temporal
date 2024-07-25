from typing import Optional

import numpy as np
from numpy.typing import NDArray

from temporal.meta.configurable import FloatParam
from temporal.meta.serializable import SerializableField as Field
from temporal.pipeline_modules.temporal import TemporalModule
from temporal.project import Project
from temporal.utils.image import NumpyImage, ensure_image_dims
from temporal.utils.math import lerp


class RandomSamplingModule(TemporalModule):
    name = "Random sampling"

    chance: float = FloatParam("Chance", minimum = 0.0, maximum = 1.0, step = 0.001, value = 1.0, ui_type = "slider")
    opacity: float = FloatParam("Opacity", minimum = 0.0, maximum = 1.0, step = 0.001, value = 1.0, ui_type = "slider")

    buffer: Optional[NDArray[np.float_]] = Field(None)

    def forward(self, images: list[NumpyImage], project: Project, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if self.buffer is None:
            self.buffer = np.stack([
                ensure_image_dims(image, "RGB", (project.processing.width, project.processing.height))
                for image in images
            ], 0)

        for i, (sub, image) in enumerate(zip(self.buffer, images)):
            size = sub.shape[:2]

            chance_mask = np.random.default_rng(seed + i).random(size) <= self.chance
            opacity_mask = np.random.default_rng(seed + 1 + i).uniform(
                low = min(max(self.opacity * 2.0 - 1.0, 0.0), 1.0),
                high = min(max(self.opacity * 2.0, 0.0), 1.0) + np.finfo(np.float_).eps,
                size = size,
            )

            sub[:] = lerp(sub, np.where(chance_mask[..., np.newaxis], image, sub), opacity_mask[..., np.newaxis])

        return [sub for sub in self.buffer]

    def reset(self) -> None:
        self.buffer = None
