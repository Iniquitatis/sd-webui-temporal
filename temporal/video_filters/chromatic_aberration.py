from typing import Iterator

from temporal.object import Param
from temporal.video_filter import VideoFilter, make_filter as mf


class ChromaticAberrationFilter(VideoFilter):
    name = "Chromatic aberration"

    distance: int = Param("Distance", minimum = 1, maximum = 512, step = 1, value = 1, ui_type = "slider")

    def generate(self, fps: int) -> Iterator[str]:
        yield mf([], [], "rgbashift", rh = -self.distance, bh = self.distance)
