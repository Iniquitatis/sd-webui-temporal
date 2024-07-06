from temporal.gradient import Gradient
from temporal.meta.configurable import GradientParam
from temporal.pipeline_modules.painting import PaintingModule
from temporal.project import Project
from temporal.utils.image import NumpyImage


class GradientPaintingModule(PaintingModule):
    name = "Gradient"

    gradient: Gradient = GradientParam("Gradient")

    def draw(self, size: tuple[int, int], parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        return self.gradient.generate((size[1], size[0], 4))
