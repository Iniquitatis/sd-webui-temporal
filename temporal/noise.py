from math import ceil
from random import randint
from typing import Any, Literal, Optional

import numpy as np
import skimage

from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.utils.numpy import FloatArray, FloatType


class Noise(Serializable):
    mode: Literal["fbm", "turbulence", "ridge"] = Field("fbm")
    scale: int = Field(1)
    detail: float = Field(1.0)
    lacunarity: float = Field(2.0)
    persistence: float = Field(0.5)
    seed: int = Field(-1)
    use_global_seed: bool = Field(False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if self.seed == -1:
            self.seed = randint(0, 0x7fffffff)

    def generate(self, shape: tuple[int, ...], global_seed: Optional[int] = None) -> FloatArray:
        noise = np.random.default_rng(
            global_seed if global_seed and self.use_global_seed else self.seed
        ).uniform(low = 0.0, high = 1.0 + np.finfo(FloatType).eps, size = shape)

        def scale_noise(scale: float) -> FloatArray:
            result = skimage.transform.warp(noise, skimage.transform.AffineTransform(scale = scale).inverse, order = 4, mode = "symmetric")

            if self.mode == "fbm":
                return result
            elif self.mode == "turbulence":
                return abs(result * 2.0 - 1.0)
            elif self.mode == "ridge":
                return 1.0 - abs(result * 2.0 - 1.0)
            else:
                raise NotImplementedError

        result = np.zeros(shape)
        total_amplitude = 0.0
        scale = self.scale
        amplitude = 0.5

        for i in range(ceil(self.detail)):
            octave_scale = min(self.detail - i, 1.0)
            result += scale_noise(scale) * (amplitude * octave_scale)
            total_amplitude += (amplitude * octave_scale)
            scale /= self.lacunarity
            amplitude *= self.persistence

        return result / total_amplitude
