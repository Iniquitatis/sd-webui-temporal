from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.tool import ToolModule
from modules.utils.fs import ensure_directory_exists
from modules.utils.image import NumpyImage, ensure_image_dims, np_to_pil, save_image
from modules.utils.math import quantize


class SavingModule(ToolModule):
    name = "Saving"

    file_name_prefix: str = Field("", name = "File name prefix", display = "box")
    scale: float = Field(1.0, name = "Scale", minimum = 0.25, maximum = 1.0, step = 0.25, display = "slider")
    archive_mode: bool = Field(False, name = "Archive mode")
    iteration: int = Field(0, flags = {"runtime"})

    def process(self, image: NumpyImage, general: GeneralData) -> None:
        save_image(
            np_to_pil(ensure_image_dims(image, size = (
                int(quantize(general.image_size.x * self.scale, 8)),
                int(quantize(general.image_size.y * self.scale, 8)),
            )) if self.scale != 1.0 else image),
            ensure_directory_exists(general.path) / f"{self.file_name_prefix}{self.iteration + 1:05d}.png",
            self.archive_mode,
        )

        self.iteration += 1
