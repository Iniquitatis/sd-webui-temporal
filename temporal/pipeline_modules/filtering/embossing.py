from math import sin, tau
from typing import Iterator

import numpy as np
from scipy import ndimage

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage, apply_channelwise
from temporal.utils.numpy import FloatArray, saturate_array


class EmbossingFilter(ImageFilter):
    name = "Embossing"

    strength: float = Field(1.0, name = "Strength", minimum = 0.01, maximum = 1.0, step = 0.01, display = "slider")
    rotation: float = Field(0.0, name = "Rotation", minimum = 0.0, maximum = 360.0, step = 1.0, suffix = "°", display = "slider")
    radius: int = Field(1, name = "Radius", minimum = 1, maximum = 5, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        def iter_edge(m: FloatArray) -> Iterator[tuple[int, int]]:
            r, c = m.shape

            yield from ((0, i) for i in range(c - 1))
            yield from ((i, c - 1) for i in range(r - 1))
            yield from ((r - 1, i) for i in range(c - 1, 0, -1))
            yield from ((i, 0) for i in range(r - 1, 0, -1))

        kernel = np.zeros((1 + self.radius * 2, 1 + self.radius * 2))

        for layer in range(self.radius):
            slice = kernel[layer:-layer, layer:-layer] if layer > 0 else kernel

            edge = list(iter_edge(slice))

            for j, (row, column) in enumerate(edge):
                slice[row, column] = sin((j / len(edge) + -self.rotation / 360.0 + 1.0 / 8.0) * tau) * self.strength

        return saturate_array(apply_channelwise(image, lambda x: ndimage.convolve(x, kernel)))
