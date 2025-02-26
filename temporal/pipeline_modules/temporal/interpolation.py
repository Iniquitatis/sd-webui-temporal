from typing import Optional

import numpy as np
import skimage

from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.meta.serializable import SerializableField as Field
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

    def forward(self, images: list[NumpyImage], general: GeneralData, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if self.buffer is None:
            self.buffer = np.stack([
                ensure_image_dims(image, (general.image_size.x, general.image_size.y), 3)
                for image in images
            ], 0)

        for sub, image in zip(self.buffer, images):
            a = sub
            b = match_image(image, sub)

            if self.movement > 0.0:
                a, b = self._motion_warp(a, b)

            sub[:] = lerp(a, b, self.blending)

        return [sub for sub in self.buffer]

    def reset(self) -> None:
        self.buffer = None

    def _motion_warp(self, base_im: NumpyImage, target_im: NumpyImage) -> tuple[NumpyImage, NumpyImage]:
        def warp(im: NumpyImage, coords: FloatArray) -> NumpyImage:
            return apply_channelwise(im, lambda x: skimage.transform.warp(x, coords, mode = "symmetric"))

        height, width = base_im.shape[:2]

        coords = np.indices((height, width)).astype(FloatType)
        offsets = skimage.registration.optical_flow_ilk(skimage.color.rgb2gray(base_im), skimage.color.rgb2gray(target_im), radius = self.radius)

        return warp(base_im, coords + offsets * -self.movement), warp(target_im, coords + -offsets * (-1.0 + self.movement))
