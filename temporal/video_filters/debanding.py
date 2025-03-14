from typing import Iterator

from temporal.object import Param
from temporal.video_filter import VideoFilter, make_filter as mf


class DebandingFilter(VideoFilter):
    name = "Debanding"

    radius: int = Param("Radius", minimum = 1, maximum = 64, step = 1, value = 16, ui_type = "slider")

    def generate(self, fps: int) -> Iterator[str]:
        yield mf([], [], "deband", range = self.radius)
