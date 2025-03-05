from typing import Iterator

from temporal.meta.configurable import ConfigurableParam as Param
from temporal.video_filter import VideoFilter, make_filter as mf


class TemporalAveragingFilter(VideoFilter):
    name = "Temporal averaging"

    radius: int = Param("Radius", minimum = 1, maximum = 60, step = 1, value = 1, ui_type = "slider")
    algorithm: str = Param("Algorithm", choices = [("mean", "Mean"), ("median", "Median")], value = "mean", ui_type = "menu")
    easing: float = Param("Easing", minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0, ui_type = "slider")

    def generate(self, fps: int) -> Iterator[str]:
        if self.algorithm == "mean":
            weights = [((x + 1) / (self.radius + 1)) ** self.easing for x in range(self.radius + 1)]
            weights += reversed(weights[:-1])
            yield mf([], [], "tmix", frames = len(weights), weights = " ".join(str(x) for x in weights))
        elif self.algorithm == "median":
            yield mf([], [], "tpad", start = self.radius, stop = self.radius, start_mode = "clone", stop_mode = "clone")
            yield mf([], [], "tmedian", radius = self.radius)
        else:
            yield mf([], [], "null")
