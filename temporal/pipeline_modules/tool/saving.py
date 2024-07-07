from typing import Optional

from temporal.meta.configurable import BoolParam, FloatParam, IntParam
from temporal.pipeline_modules.tool import ToolModule
from temporal.project import Project
from temporal.shared import shared
from temporal.utils.fs import ensure_directory_exists
from temporal.utils.image import NumpyImage, ensure_image_dims, np_to_pil
from temporal.utils.math import quantize
from temporal.utils.time import wait_until


class SavingModule(ToolModule):
    name = "Saving"

    scale: float = FloatParam("Scale", minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0, ui_type = "slider")
    save_every_nth_frame: int = IntParam("Save every N-th frame", minimum = 1, step = 1, value = 1, ui_type = "box")
    save_final: bool = BoolParam("Save final", value = False)
    archive_mode: bool = BoolParam("Archive mode", value = False)

    def forward(self, images: list[NumpyImage], project: Project, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if frame_index % self.save_every_nth_frame != 0:
            return images

        for i, image in enumerate(self._get_scaled_images(images, project)):
            file_name = f"{frame_index:05d}"

            if len(images) > 1:
                file_name += f"-{i + 1:02d}"

            shared.backend.save_image(
                image = np_to_pil(image),
                output_dir = ensure_directory_exists(project.path),
                file_name = file_name,
                archive_mode = self.archive_mode,
            )

        return images

    def finalize(self, images: list[NumpyImage], project: Project) -> None:
        if self.save_final:
            for image in self._get_scaled_images(images, project):
                shared.backend.save_image(
                    image = np_to_pil(image),
                    output_dir = ensure_directory_exists(shared.options.output.output_dir),
                    file_name = None,
                    archive_mode = self.archive_mode,
                )

        wait_until(lambda: shared.backend.is_done_saving)

    def _get_scaled_images(self, images: list[NumpyImage], project: Project) -> list[NumpyImage]:
        return [ensure_image_dims(x, size = (
            int(quantize(project.backend_data.width * self.scale, 8)),
            int(quantize(project.backend_data.height * self.scale, 8)),
        )) for x in images] if self.scale != 1.0 else images
