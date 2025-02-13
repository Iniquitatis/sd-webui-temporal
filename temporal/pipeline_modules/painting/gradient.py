from temporal.general_data import GeneralData
from temporal.gradient import Gradient
from temporal.meta.configurable import GradientParam
from temporal.pipeline_modules.painting import PaintingModule
from temporal.utils.image import NumpyImage


class GradientPaintingModule(PaintingModule):
    name = "Gradient"

    gradient: Gradient = GradientParam("Gradient")

    def draw(self, size: tuple[int, int], parallel_index: int, general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        return self.gradient.generate((size[1], size[0], 4))
