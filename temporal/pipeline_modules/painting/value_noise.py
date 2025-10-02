from math import ceil

import numpy as np
import skimage

from temporal.color import Color
from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.painting import PaintingModule
from temporal.seed import Seed
from temporal.utils.image import NumpyImage, make_trs_transform
from temporal.utils.math import lerp
from temporal.utils.numpy import FloatArray, FloatType, random_array


class ValueNoisePaintingModule(PaintingModule):
    name = "Value noise"

    type: str = Field("duochrome", name = "Type", choices = {"duochrome": "Duochrome", "colored": "Colored"}, display = "radio")
    mode: str = Field("fbm", name = "Mode", choices = {"fbm": "fBm", "turbulence": "Turbulence", "ridge": "Ridge"}, display = "radio")
    scale: int = Field(1, name = "Scale", minimum = 1, maximum = 1024, step = 1, suffix = " px", display = "slider")
    detail: float = Field(1.0, name = "Detail", minimum = 1.0, maximum = 10.0, step = 0.01, display = "slider")
    lacunarity: float = Field(2.0, name = "Lacunarity", minimum = 0.01, maximum = 4.0, step = 0.01, display = "slider")
    persistence: float = Field(0.5, name = "Persistence", minimum = 0.0, maximum = 1.0, step = 0.01, display = "slider")
    use_global_seed: bool = Field(False, name = "Use global seed")
    seed: Seed = Field(Seed, name = "Seed", dependencies = {"use_global_seed": False})
    advance_seed: bool = Field(False, name = "Advance seed")
    color_a: Color = Field(lambda: Color(0.0, 0.0, 0.0), name = "Color A", channels = 4, dependencies = {"type": "duochrome"})
    color_b: Color = Field(lambda: Color(1.0, 1.0, 1.0), name = "Color B", channels = 4, dependencies = {"type": "duochrome"})
    iteration: int = Field(0, flags = {"runtime"})

    def draw(self, size: tuple[int, int], general: GeneralData) -> NumpyImage:
        seed = (general.seed if self.use_global_seed else self.seed).fixed_value

        if self.advance_seed:
            seed += self.iteration

        if self.type == "duochrome":
            result = lerp(
                self.color_a.to_numpy(4),
                self.color_b.to_numpy(4),
                self._generate((size[1], size[0], 1), seed),
            )
        elif self.type == "colored":
            result = self._generate((size[1], size[0], 3), seed)
        else:
            raise ValueError(f"Incorrect type {self.type}")

        self.iteration += 1

        return result

    def _generate(self, shape: tuple[int, ...], seed: int) -> FloatArray:
        octave_count = ceil(self.detail)

        noises = random_array((octave_count,) + shape, low = 0.0, high = 1.0, seed = seed)

        result = np.zeros(shape, dtype = FloatType)
        total_amplitude = 0.0
        scale = self.scale
        amplitude = 0.5

        for i in range(octave_count):
            noise = skimage.transform.warp(
                noises[i],
                make_trs_transform(image_size = (shape[1], shape[0]), scale = scale),
                order = 4,
                mode = "symmetric",
            )

            if self.mode == "fbm":
                pass
            elif self.mode == "turbulence":
                noise = abs(noise * 2.0 - 1.0)
            elif self.mode == "ridge":
                noise = 1.0 - abs(noise * 2.0 - 1.0)
            else:
                raise NotImplementedError

            octave_scale = min(self.detail - i, 1.0)
            contribution = amplitude * octave_scale

            result += noise * contribution
            total_amplitude += contribution
            scale /= self.lacunarity
            amplitude *= self.persistence

        result /= total_amplitude

        return result
