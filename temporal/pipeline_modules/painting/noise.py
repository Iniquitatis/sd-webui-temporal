import numpy as np

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.painting import PaintingModule
from temporal.seed import Seed
from temporal.utils.image import NumpyImage
from temporal.utils.numpy import FloatType


class NoisePaintingModule(PaintingModule):
    name = "Noise"

    type: str = Field("uniform", name = "Type", choices = {"uniform": "Uniform", "gaussian": "Gaussian", "poisson": "Poisson"}, display = "radio")
    minimum: float = Field(0.0, name = "Minimum", minimum = 0.0, maximum = 1.0, step = 0.01, dependencies = {"type": ["uniform", "poisson"]}, display = "slider")
    maximum: float = Field(1.0, name = "Maximum", minimum = 0.0, maximum = 1.0, step = 0.01, dependencies = {"type": ["uniform", "poisson"]}, display = "slider")
    mean: float = Field(0.5, name = "Mean", minimum = 0.0, maximum = 1.0, step = 0.01, dependencies = {"type": "gaussian"}, display = "slider")
    standard_deviation: float = Field(0.5, name = "Standard deviation", minimum = 0.0, maximum = 1.0, step = 0.01, dependencies = {"type": "gaussian"}, display = "slider")
    intensity: float = Field(0.5, name = "Intensity", minimum = 0.0, maximum = 1.0, step = 0.01, dependencies = {"type": "poisson"}, display = "slider")
    colored: bool = Field(True, name = "Colored")
    alpha: bool = Field(False, name = "Alpha")
    use_global_seed: bool = Field(False, name = "Use global seed")
    seed: Seed = Field(Seed, name = "Seed", dependencies = {"use_global_seed": False})
    advance_seed: bool = Field(False, name = "Advance seed")
    iteration: int = Field(0, flags = {"runtime"})

    def draw(self, size: tuple[int, int], general: GeneralData) -> NumpyImage:
        seed = (general.seed if self.use_global_seed else self.seed).fixed_value

        if self.advance_seed:
            seed += self.iteration

        rng = np.random.default_rng(seed)

        if self.colored and self.alpha:
            channels = 4
        elif self.colored:
            channels = 3
        elif self.alpha:
            channels = 2
        else:
            channels = 1

        shape = size[1], size[0], channels

        if self.type == "uniform":
            noise = self.minimum + rng.random(shape, dtype = FloatType) * (self.maximum - self.minimum)
        elif self.type == "gaussian":
            noise = self.mean + rng.standard_normal(shape, dtype = FloatType) * self.standard_deviation
        elif self.type == "poisson" and self.intensity > 0.0:
            noise = self.minimum + rng.poisson(self.intensity, shape).astype(FloatType) / self.intensity * (self.maximum - self.minimum)
        elif self.type == "poisson" and self.intensity == 0.0:
            noise = self.minimum + np.zeros(shape) * (self.maximum - self.minimum)
        else:
            raise ValueError(f"Incorrect type {self.type}")

        if self.colored and self.alpha:
            result = noise
        elif self.colored:
            result = noise
        elif self.alpha:
            result = np.stack([noise[..., 0], noise[..., 0], noise[..., 0], noise[..., 1]], axis = -1)
        else:
            result = np.stack([noise[..., 0], noise[..., 0], noise[..., 0]], axis = -1)

        self.iteration += 1

        return result
