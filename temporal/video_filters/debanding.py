from typing import Iterator

from temporal.object import Field
from temporal.video_filter import VideoFilter, make_filter as mf


class DebandingFilter(VideoFilter):
    name = "Debanding"

    radius: int = Field(16, name = "Radius", minimum = 1, maximum = 64, step = 1, display = "slider")

    def generate(self, fps: int) -> Iterator[str]:
        yield mf([], [], "deband", range = self.radius)
