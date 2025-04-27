from typing import Iterator

from temporal.object import Field
from temporal.video_filter import VideoFilter, make_filter as mf


class ColorBalancingFilter(VideoFilter):
    name = "Color balancing"

    brightness: float = Field(1.0, name = "Brightness", minimum = 0.0, maximum = 2.0, step = 0.01, display = "slider")
    contrast: float = Field(1.0, name = "Contrast", minimum = 0.0, maximum = 2.0, step = 0.01, display = "slider")
    saturation: float = Field(1.0, name = "Saturation", minimum = 0.0, maximum = 2.0, step = 0.01, display = "slider")

    def generate(self, fps: int) -> Iterator[str]:
        yield mf([], [], "eq",
            contrast = self.contrast,
            brightness = self.brightness - 1.0,
            saturation = self.saturation,
        )
