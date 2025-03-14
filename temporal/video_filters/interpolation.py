from typing import Iterator

from temporal.object import Param
from temporal.video_filter import VideoFilter, make_filter as mf


class InterpolationFilter(VideoFilter):
    name = "Interpolation"

    fps: int = Param("Frames per second", minimum = 1, maximum = 60, step = 1, value = 60, ui_type = "slider")
    mb_subframes: int = Param("Motion blur subframes", minimum = 0, maximum = 15, step = 1, value = 0, ui_type = "slider")

    def generate(self, fps: int) -> Iterator[str]:
        yield mf([], [], "minterpolate",
            fps = self.fps * (self.mb_subframes + 1),
            mi_mode = "mci",
            mc_mode = "aobmc",
            me_mode = "bidir",
            vsbmc = 1,
            scd = "none",
        )

        if self.mb_subframes > 0:
            yield mf([], [], "tmix", frames = self.mb_subframes + 1)
            yield mf([], [], "fps", self.fps)
