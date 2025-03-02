from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.pattern import Pattern
from temporal.pipeline_modules.painting import PaintingModule
from temporal.utils.image import NumpyImage


class PatternPaintingModule(PaintingModule):
    name = "Pattern"

    pattern: Pattern = Param("Pattern", factory = Pattern)

    def draw(self, size: tuple[int, int], general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        return self.pattern.generate((size[1], size[0], 4))
