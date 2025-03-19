from typing import Optional

import numpy as np
import skimage

from temporal.general_data import GeneralData
from temporal.object import Field, Param
from temporal.pipeline_modules.temporal import TemporalModule
from temporal.utils.image import NumpyImage, apply_channelwise, ensure_image_dims, match_image
from temporal.utils.math import lerp
from temporal.utils.numpy import FloatArray, FloatType


class InterpolationModule(TemporalModule):
    name = "Interpolation"

    blending: float = Param("Blending", minimum = 0.0, maximum = 1.0, step = 0.001, value = 1.0, ui_type = "slider")
    movement: float = Param("Movement", minimum = 0.0, maximum = 1.0, step = 0.001, value = 1.0, ui_type = "slider")
    radius: int = Param("Radius", minimum = 7, maximum = 31, step = 2, value = 15, ui_type = "slider")

    buffer: Optional[FloatArray] = Field(None, flags = {"private"})

    def forward(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> Optional[NumpyImage]:
        if self.buffer is None:
            self.buffer = ensure_image_dims(image.copy(), (general.image_size.x, general.image_size.y), 3)

        a = self.buffer
        b = match_image(image, self.buffer)

        if self.movement > 0.0:
            a, b = self._motion_warp(a, b)

        self.buffer[:] = lerp(a, b, self.blending)

        return self.buffer.copy()

    def reset(self, general: GeneralData) -> None:
        self.buffer = None

    def _motion_warp(self, base: NumpyImage, target: NumpyImage) -> tuple[NumpyImage, NumpyImage]:
        def warp(image: NumpyImage, coords: FloatArray) -> NumpyImage:
            return apply_channelwise(image, lambda x: skimage.transform.warp(x, coords, mode = "symmetric"))

        height, width = base.shape[:2]

        coords = np.indices((height, width)).astype(FloatType)
        offsets = skimage.registration.optical_flow_ilk(skimage.color.rgb2gray(base), skimage.color.rgb2gray(target), radius = self.radius)

        return warp(base, coords + offsets * -self.movement), warp(target, coords + -offsets * (-1.0 + self.movement))
