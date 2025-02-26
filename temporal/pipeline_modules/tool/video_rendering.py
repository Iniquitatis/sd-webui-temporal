from typing import Optional

from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.shared import shared
from temporal.pipeline_modules.tool import ToolModule
from temporal.utils.image import NumpyImage
from temporal.utils.time import wait_until
from temporal.video_renderer import video_render_queue


class VideoRenderingModule(ToolModule):
    name = "Video rendering"

    render_draft_every_nth_frame: int = Param("Render draft every N-th frame", minimum = 1, step = 1, value = 100, ui_type = "box")
    render_final_every_nth_frame: int = Param("Render final every N-th frame", minimum = 1, step = 1, value = 1000, ui_type = "box")
    render_draft_on_finish: bool = Param("Render draft on finish", value = False)
    render_final_on_finish: bool = Param("Render final on finish", value = False)

    def forward(self, images: list[NumpyImage], general: GeneralData, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        for i, _ in enumerate(images, 1):
            if frame_index % self.render_draft_every_nth_frame == 0:
                general.render_video(shared.video_renderer, False, i)

            if frame_index % self.render_final_every_nth_frame == 0:
                general.render_video(shared.video_renderer, True, i)

        return images

    def finalize(self, images: list[NumpyImage], general: GeneralData) -> None:
        for i, _ in enumerate(images, 1):
            if self.render_draft_on_finish:
                general.render_video(shared.video_renderer, False, i)

            if self.render_final_on_finish:
                general.render_video(shared.video_renderer, True, i)

        wait_until(lambda: not video_render_queue.busy)
