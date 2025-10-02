from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.tool import ToolModule
from temporal.thread_queue import ThreadQueue
from temporal.utils.fs import ensure_directory_exists
from temporal.utils.image import NumpyImage, ensure_image_dims, np_to_pil, save_image
from temporal.utils.math import quantize
from temporal.utils.time import wait_until


class SavingModule(ToolModule):
    name = "Saving"

    file_name_prefix: str = Field("", name = "File name prefix", display = "box")
    scale: float = Field(1.0, name = "Scale", minimum = 0.25, maximum = 1.0, step = 0.25, display = "slider")
    save_every_nth_iteration: int = Field(1, name = "Save every N-th iteration", minimum = 1, step = 1, display = "box")
    archive_mode: bool = Field(False, name = "Archive mode")
    iteration: int = Field(0, flags = {"runtime"})

    def process(self, image: NumpyImage, general: GeneralData) -> None:
        if self.iteration % self.save_every_nth_iteration == 0:
            _save_queue.enqueue(
                save_image,
                np_to_pil(ensure_image_dims(image, size = (
                    int(quantize(general.image_size.x * self.scale, 8)),
                    int(quantize(general.image_size.y * self.scale, 8)),
                )) if self.scale != 1.0 else image),
                ensure_directory_exists(general.path) / f"{self.file_name_prefix}{self.iteration + 1:05d}.png",
                self.archive_mode,
            )

        self.iteration += 1

    def finalize(self, general: GeneralData) -> None:
        wait_until(lambda: not _save_queue.busy)


_save_queue = ThreadQueue()
