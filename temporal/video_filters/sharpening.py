from typing import Iterator

from temporal.object import Param
from temporal.video_filter import VideoFilter, make_filter as mf


class SharpeningFilter(VideoFilter):
    name = "Sharpening"

    strength: float = Param("Strength", minimum = 0.0, maximum = 1.0, step = 0.1, value = 0.0, ui_type = "slider")
    radius: int = Param("Radius", minimum = 3, maximum = 13, step = 2, value = 3, ui_type = "slider")

    def generate(self, fps: int) -> Iterator[str]:
        yield mf([], [], "unsharp",
            luma_msize_x = self.radius,
            luma_msize_y = self.radius,
            luma_amount = self.strength,
            chroma_msize_x = self.radius,
            chroma_msize_y = self.radius,
            chroma_amount = self.strength,
        )
