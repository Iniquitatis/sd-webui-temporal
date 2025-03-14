from typing import Iterator

from temporal.object import Param
from temporal.video_filter import VideoFilter, make_filter as mf


class DeflickeringFilter(VideoFilter):
    name = "Deflickering"

    frames: int = Param("Frames", minimum = 2, maximum = 120, step = 1, value = 60, ui_type = "slider")

    def generate(self, fps: int) -> Iterator[str]:
        yield mf([], [], "deflicker", size = self.frames, mode = "am")
