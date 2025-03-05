from pathlib import Path

from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.pipeline_modules.tool import ToolModule
from temporal.utils.fs import ensure_directory_exists, remove_entry
from temporal.utils.image import NumpyImage
from temporal.utils.time import wait_until
from temporal.video_renderer import VideoRenderer, video_render_queue


class VideoRenderingModule(ToolModule):
    name = "Video rendering"

    file_name: str = Param("File name", value = "video", ui_type = "box")
    image_name_prefix: str = Param("Image name prefix", value = "", ui_type = "box")
    render_every_nth_frame: int = Param("Render every N-th frame", minimum = 1, step = 1, value = 100, ui_type = "box")
    renderer: VideoRenderer = Param("Rendering parameters", factory = VideoRenderer)

    def process(self, npim: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> None:
        if frame_index % self.render_every_nth_frame == 0:
            self.render(general, True)

    def finalize(self, image: NumpyImage, general: GeneralData) -> None:
        wait_until(lambda: not video_render_queue.busy)

    def reset(self, general: GeneralData) -> None:
        remove_entry(general.path / "videos" / f"{self.file_name}.mp4")

    def render(self, general: GeneralData, enqueue: bool) -> Path:
        path = ensure_directory_exists(general.path / "videos") / f"{self.file_name}.mp4"

        self.renderer.render(path, sorted(
            general.path.glob(f"{self.image_name_prefix}*.png"),
            key = lambda x: x.name,
        ), enqueue)

        return path
