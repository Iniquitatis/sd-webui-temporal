from temporal.meta.configurable import PatternParam
from temporal.pattern import Pattern
from temporal.pipeline_modules.painting import PaintingModule
from temporal.project import Project
from temporal.utils.image import NumpyImage


class PatternPaintingModule(PaintingModule):
    id = "pattern_painting"
    name = "Pattern"

    pattern: Pattern = PatternParam("Pattern")

    def draw(self, size: tuple[int, int], parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        return self.pattern.generate((size[1], size[0], 4))
