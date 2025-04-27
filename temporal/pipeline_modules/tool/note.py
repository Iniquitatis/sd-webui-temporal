from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.tool import ToolModule
from temporal.utils.image import NumpyImage


class NoteModule(ToolModule):
    name = "Note"

    text: str = Field("", name = "Text", display = "area")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> None:
        pass
