from typing import Iterator

from temporal.meta.configurable import ConfigurableParam as Param
from temporal.video_filter import VideoFilter, make_filter as mf


class ColorBalancingFilter(VideoFilter):
    name = "Color balancing"

    brightness: float = Param("Brightness", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, ui_type = "slider")
    contrast: float = Param("Contrast", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, ui_type = "slider")
    saturation: float = Param("Saturation", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, ui_type = "slider")

    def generate(self, fps: int) -> Iterator[str]:
        yield mf([], [], "eq",
            contrast = self.contrast,
            brightness = self.brightness - 1.0,
            saturation = self.saturation,
        )
