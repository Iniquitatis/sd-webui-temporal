from math import ceil
from random import randint
from typing import Any, Optional

import numpy as np
import skimage

from temporal.color import Color
from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.pipeline_modules.painting import PaintingModule
from temporal.utils.image import NumpyImage, make_trs_transform
from temporal.utils.math import lerp
from temporal.utils.numpy import FloatArray, random_array


class NoisePaintingModule(PaintingModule):
    name = "Noise"

    type: str = Param("Type", choices = {"duochrome": "Duochrome", "colored": "Colored"}, value = "duochrome", ui_type = "radio")
    mode: str = Param("Mode", choices = {"fbm": "fBm", "turbulence": "Turbulence", "ridge": "Ridge"}, value = "fbm", ui_type = "radio")
    scale: int = Param("Scale", minimum = 1, maximum = 1024, step = 1, value = 1, ui_type = "slider")
    detail: float = Param("Detail", minimum = 1.0, maximum = 10.0, step = 0.01, value = 1.0, ui_type = "slider")
    lacunarity: float = Param("Lacunarity", minimum = 0.01, maximum = 4.0, step = 0.01, value = 2.0, ui_type = "slider")
    persistence: float = Param("Persistence", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.5, ui_type = "slider")
    seed: int = Param("Seed", value = -1, ui_type = "seed")
    use_global_seed: bool = Param("Use global seed", value = False)
    color_a: Color = Param("Color A", value = lambda: Color(0.0, 0.0, 0.0), channels = 4, dependencies = {"type": "duochrome"})
    color_b: Color = Param("Color B", value = lambda: Color(1.0, 1.0, 1.0), channels = 4, dependencies = {"type": "duochrome"})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if self.seed == -1:
            self.seed = randint(0, 0x7fffffff)

    def draw(self, size: tuple[int, int], general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        if self.type == "duochrome":
            return lerp(
                self.color_a.to_numpy(4),
                self.color_b.to_numpy(4),
                self._generate((size[1], size[0], 1), seed),
            )
        elif self.type == "colored":
            return self._generate((size[1], size[0], 3), seed)
        else:
            raise ValueError(f"Incorrect type {self.type}")

    def _generate(self, shape: tuple[int, ...], global_seed: Optional[int] = None) -> FloatArray:
        noises = random_array(
            (ceil(self.detail),) + shape,
            low = 0.0,
            high = 1.0,
            seed = global_seed if global_seed and self.use_global_seed else self.seed,
        )

        def scale_noise(i: int, scale: float) -> FloatArray:
            result = skimage.transform.warp(noises[i], make_trs_transform(image_size = (shape[1], shape[0]), scale = scale), order = 4, mode = "symmetric")

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
            result += scale_noise(i, scale) * (amplitude * octave_scale)
            total_amplitude += (amplitude * octave_scale)
            scale /= self.lacunarity
            amplitude *= self.persistence

        return result / total_amplitude
