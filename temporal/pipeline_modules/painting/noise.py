from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.noise import Noise
from temporal.pipeline_modules.painting import PaintingModule
from temporal.utils.image import NumpyImage


class NoisePaintingModule(PaintingModule):
    name = "Noise"

    noise: Noise = Param("Noise", factory = Noise)

    def draw(self, size: tuple[int, int], parallel_index: int, general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        return self.noise.generate((size[1], size[0], 3), seed)
