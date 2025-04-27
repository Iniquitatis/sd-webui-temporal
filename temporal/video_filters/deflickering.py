from typing import Iterator

from temporal.object import Field
from temporal.video_filter import VideoFilter, make_filter as mf


class DeflickeringFilter(VideoFilter):
    name = "Deflickering"

    frames: int = Field(60, name = "Frames", minimum = 2, maximum = 120, step = 1, display = "slider")

    def generate(self, fps: int) -> Iterator[str]:
        yield mf([], [], "deflicker", size = self.frames, mode = "am")
