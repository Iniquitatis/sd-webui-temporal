from math import ceil
from typing import Literal, Optional

import numpy as np
import skimage
from numpy.typing import NDArray

from temporal.meta.serializable import Serializable, SerializableField as Field


class Noise(Serializable):
    mode: Literal["fbm", "turbulence", "ridge"] = Field("fbm")
    scale: int = Field(1)
    detail: float = Field(1.0)
    lacunarity: float = Field(2.0)
    persistence: float = Field(0.5)
    seed: int = Field(0)
    use_global_seed: bool = Field(False)

    def generate(self, shape: tuple[int, ...], global_seed: Optional[int] = None, seed_offset: int = 0) -> NDArray[np.float_]:
        noise = np.random.default_rng(
            (global_seed if global_seed and self.use_global_seed else self.seed) + seed_offset
        ).uniform(low = 0.0, high = 1.0 + np.finfo(np.float_).eps, size = shape)

        def scale_noise(scale: float) -> NDArray[np.float_]:
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
