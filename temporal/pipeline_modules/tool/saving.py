from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.pipeline_modules.tool import ToolModule
from temporal.thread_queue import ThreadQueue
from temporal.utils.fs import clear_directory, ensure_directory_exists
from temporal.utils.image import NumpyImage, ensure_image_dims, np_to_pil, save_image
from temporal.utils.math import quantize
from temporal.utils.time import wait_until


class SavingModule(ToolModule):
    name = "Saving"

    file_name_prefix: str = Param("File name prefix", value = "", ui_type = "box")
    scale: float = Param("Scale", minimum = 0.25, maximum = 1.0, step = 0.25, value = 1.0, ui_type = "slider")
    save_every_nth_frame: int = Param("Save every N-th frame", minimum = 1, step = 1, value = 1, ui_type = "box")
    archive_mode: bool = Param("Archive mode", value = False)

    def process(self, image: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> None:
        if frame_index % self.save_every_nth_frame == 0:
            _save_queue.enqueue(
                save_image,
                np_to_pil(ensure_image_dims(image, size = (
                    int(quantize(general.image_size.x * self.scale, 8)),
                    int(quantize(general.image_size.y * self.scale, 8)),
                )) if self.scale != 1.0 else image),
                ensure_directory_exists(general.path) / f"{self.file_name_prefix}{frame_index:05d}.png",
                self.archive_mode,
            )

    def finalize(self, image: NumpyImage, general: GeneralData) -> None:
        wait_until(lambda: not _save_queue.busy)

    def reset(self, general: GeneralData) -> None:
        clear_directory(general.path, f"{self.file_name_prefix}*.png")


_save_queue = ThreadQueue()
