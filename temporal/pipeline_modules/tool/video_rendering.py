from typing import Optional

from temporal.meta.configurable import BoolParam, IntParam
from temporal.project import Project
from temporal.shared import shared
from temporal.pipeline_modules.tool import ToolModule
from temporal.utils.image import NumpyImage
from temporal.utils.time import wait_until
from temporal.video_renderer import video_render_queue


class VideoRenderingModule(ToolModule):
    name = "Video rendering"

    render_draft_every_nth_frame: int = IntParam("Render draft every N-th frame", minimum = 1, step = 1, value = 100, ui_type = "box")
    render_final_every_nth_frame: int = IntParam("Render final every N-th frame", minimum = 1, step = 1, value = 1000, ui_type = "box")
    render_draft_on_finish: bool = BoolParam("Render draft on finish", value = False)
    render_final_on_finish: bool = BoolParam("Render final on finish", value = False)

    def forward(self, images: list[NumpyImage], project: Project, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        for i, _ in enumerate(images, 1):
            if frame_index % self.render_draft_every_nth_frame == 0:
                project.render_video(shared.video_renderer, False, i)

            if frame_index % self.render_final_every_nth_frame == 0:
                project.render_video(shared.video_renderer, True, i)

        return images

    def finalize(self, images: list[NumpyImage], project: Project) -> None:
        for i, _ in enumerate(images, 1):
            if self.render_draft_on_finish:
                project.render_video(shared.video_renderer, False, i)

            if self.render_final_on_finish:
                project.render_video(shared.video_renderer, True, i)

        wait_until(lambda: not video_render_queue.busy)
