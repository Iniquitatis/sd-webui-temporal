from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.tool import ToolModule
from modules.utils.image import NumpyImage


class NoteModule(ToolModule):
    name = "Note"

    text: str = Field("", name = "Text", display = "area")

    def process(self, image: NumpyImage, general: GeneralData) -> None:
        pass
