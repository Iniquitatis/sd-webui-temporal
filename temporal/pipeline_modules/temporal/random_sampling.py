from typing import Optional

import numpy as np
from numpy.typing import NDArray

from temporal.meta.configurable import FloatParam
from temporal.meta.serializable import SerializableField as Field
from temporal.pipeline_modules.temporal import TemporalModule
from temporal.project import Project
from temporal.utils.image import NumpyImage, ensure_image_dims


class RandomSamplingModule(TemporalModule):
    name = "Random sampling"

    chance: float = FloatParam("Chance", minimum = 0.001, maximum = 1.0, step = 0.001, value = 1.0, ui_type = "slider")

    buffer: Optional[NDArray[np.float_]] = Field(None)

    def forward(self, images: list[NumpyImage], project: Project, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if self.buffer is None:
            self.buffer = np.stack([
                ensure_image_dims(image, "RGB", (project.backend_data.width, project.backend_data.height))
                for image in images
            ], 0)

        for i, (sub, image) in enumerate(zip(self.buffer, images)):
            mask = np.random.default_rng(seed + i).random(sub.shape[:2]) <= self.chance

            for j in range(sub.shape[-1]):
                sub[..., j] = np.where(mask, image[..., j], sub[..., j])

        return [sub for sub in self.buffer]

    def reset(self) -> None:
        self.buffer = None
