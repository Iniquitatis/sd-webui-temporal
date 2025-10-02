from typing import Iterator

from modules.object import Field
from modules.video_filter import VideoFilter, make_filter as mf


class ChromaticAberrationFilter(VideoFilter):
    name = "Chromatic aberration"

    distance: int = Field(1, name = "Distance", minimum = 1, maximum = 512, step = 1, display = "slider")

    def generate(self, fps: int) -> Iterator[str]:
        yield mf([], [], "rgbashift", rh = -self.distance, bh = self.distance)
