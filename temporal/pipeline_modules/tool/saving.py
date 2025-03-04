from typing import Optional

from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.pipeline_modules.tool import ToolModule
from temporal.shared import shared
from temporal.utils.fs import ensure_directory_exists
from temporal.utils.image import NumpyImage, ensure_image_dims
from temporal.utils.math import quantize
from temporal.utils.time import wait_until


class SavingModule(ToolModule):
    name = "Saving"

    scale: float = Param("Scale", minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0, ui_type = "slider")
    save_every_nth_frame: int = Param("Save every N-th frame", minimum = 1, step = 1, value = 1, ui_type = "box")
    save_final: bool = Param("Save final", value = False)
    archive_mode: bool = Param("Archive mode", value = False)

    def forward(self, image: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> Optional[NumpyImage]:
        if frame_index % self.save_every_nth_frame != 0:
            return image

        shared.backend.save_image(
            image = self._get_scaled_image(image, general),
            general = general,
            output_dir = ensure_directory_exists(general.path),
            file_name = f"{frame_index:05d}",
            archive_mode = self.archive_mode,
        )

        return image

    def finalize(self, image: NumpyImage, general: GeneralData) -> None:
        if self.save_final:
            shared.backend.save_image(
                image = self._get_scaled_image(image, general),
                general = general,
                output_dir = ensure_directory_exists(shared.settings.output.output_dir),
                file_name = None,
                archive_mode = self.archive_mode,
            )

        wait_until(shared.backend.are_images_saved)

    def _get_scaled_image(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        return ensure_image_dims(image, size = (
            int(quantize(general.image_size.x * self.scale, 8)),
            int(quantize(general.image_size.y * self.scale, 8)),
        )) if self.scale != 1.0 else image
