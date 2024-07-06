from temporal.meta.configurable import NoiseParam
from temporal.noise import Noise
from temporal.pipeline_modules.painting import PaintingModule
from temporal.project import Project
from temporal.utils.image import NumpyImage


class NoisePaintingModule(PaintingModule):
    name = "Noise"

    noise: Noise = NoiseParam("Noise")

    def draw(self, size: tuple[int, int], parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        return self.noise.generate((size[1], size[0], 3), seed)
