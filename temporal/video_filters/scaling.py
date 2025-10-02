from typing import Iterator

from modules.color import Color
from modules.object import Field
from modules.vector import IntVector
from modules.video_filter import VideoFilter, make_filter as mf


class ScalingFilter(VideoFilter):
    name = "Scaling"

    size: IntVector = Field(lambda: IntVector(512, 512), name = "Size", axes = ["Width", "Height"], minimum = 16, maximum = 2560, step = 8, display = "slider")
    padded: bool = Field(False, name = "Padded")
    background_color: Color = Field(lambda: Color(0.0, 0.0, 0.0), name = "Background color", channels = 3)
    backdrop: bool = Field(False, name = "Backdrop")
    backdrop_brightness: float = Field(0.5, name = "Backdrop brightness", minimum = 0.0, maximum = 2.0, step = 0.01, display = "slider")
    backdrop_blurring: float = Field(0.0, name = "Backdrop blurring", minimum = 0.0, maximum = 50.0, step = 1.0, display = "slider")

    def generate(self, fps: int) -> Iterator[str]:
        if self.padded:
            yield mf([], ["bg", "fg"], "split")

            if self.size.x > self.size.y:
                bgsw, bgsh = self.size.x, -1
                fgsw, fgsh = -1, self.size.y
                pw, ph = self.size.x, "ih"
                px, py = "(ow-iw)/2", 0
            else:
                bgsw, bgsh = -1, self.size.y
                fgsw, fgsh = self.size.x, -1
                pw, ph = "iw", self.size.y
                px, py = 0, "(oh-ih)/2"

            if self.backdrop:
                yield mf(["bg"], ["bg"], "scale", w = bgsw, h = bgsh, flags = "lanczos")
                yield mf(["bg"], ["bg"], "crop", w = self.size.x, h = self.size.y)
                yield mf(["bg"], ["bg"], "eq", brightness = self.backdrop_brightness - 1.0)
                yield mf(["bg"], ["bg"], "gblur", sigma = self.backdrop_blurring)
            else:
                yield mf(["bg"], ["bg"], "scale", w = self.size.x, h = self.size.y, flags = "neighbor")
                yield mf(["bg"], ["bg"], "drawbox", w = self.size.x, h = self.size.y, color = self.background_color.to_hex(), thickness = "fill")

            yield mf(["fg"], ["fg"], "scale", w = fgsw, h = fgsh, flags = "lanczos")
            yield mf(["fg"], ["fg"], "pad", w = pw, h = ph, x = px, y = py, color = "#00000000")
            yield mf(["bg", "fg"], [], "overlay")
        else:
            yield mf([], [], "scale", self.size.x, self.size.y, flags = "lanczos")
