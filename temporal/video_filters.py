from abc import abstractmethod
from typing import Iterator, Type

from temporal.color import Color
from temporal.meta.configurable import Configurable, ConfigurableParam as Param
from temporal.meta.serializable import SerializableField as Field
from temporal.vector import FloatVector, IntVector


VIDEO_FILTERS: list[Type["VideoFilter"]] = []


class VideoFilter(Configurable, abstract = True):
    store = VIDEO_FILTERS

    enabled: bool = Field(True)

    def print(self, fps: int) -> str:
        return ",".join(self.generate(fps))

    @abstractmethod
    def generate(self, fps: int) -> Iterator[str]:
        raise NotImplementedError


class ChromaticAberrationFilter(VideoFilter):
    name = "Chromatic aberration"

    distance: int = Param("Distance", minimum = 1, maximum = 512, step = 1, value = 1, ui_type = "slider")

    def generate(self, fps: int) -> Iterator[str]:
        yield mf([], [], "rgbashift", rh = -self.distance, bh = self.distance)


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


class DebandingFilter(VideoFilter):
    name = "Debanding"

    radius: int = Param("Radius", minimum = 1, maximum = 64, step = 1, value = 16, ui_type = "slider")

    def generate(self, fps: int) -> Iterator[str]:
        yield mf([], [], "deband", range = self.radius)


class DeflickeringFilter(VideoFilter):
    name = "Deflickering"

    frames: int = Param("Frames", minimum = 2, maximum = 120, step = 1, value = 60, ui_type = "slider")

    def generate(self, fps: int) -> Iterator[str]:
        yield mf([], [], "deflicker", size = self.frames, mode = "am")


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


class ScalingFilter(VideoFilter):
    name = "Scaling"

    size: IntVector = Param("Size", axes = ["Width", "Height"], minimum = 16, maximum = 2560, step = 8, factory = lambda: IntVector(512, 512), ui_type = "slider")
    padded: bool = Param("Padded", value = False)
    background_color: Color = Param("Background color", channels = 3, factory = lambda: Color(0.0, 0.0, 0.0))
    backdrop: bool = Param("Backdrop", value = False)
    backdrop_brightness: float = Param("Backdrop brightness", minimum = 0.0, maximum = 2.0, step = 0.01, value = 0.5, ui_type = "slider")
    backdrop_blurring: float = Param("Backdrop blurring", minimum = 0.0, maximum = 50.0, step = 1.0, value = 0.0, ui_type = "slider")

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


class SharpeningFilter(VideoFilter):
    name = "Sharpening"

    strength: float = Param("Strength", minimum = 0.0, maximum = 1.0, step = 0.1, value = 0.0, ui_type = "slider")
    radius: int = Param("Radius", minimum = 3, maximum = 13, step = 2, value = 3, ui_type = "slider")

    def generate(self, fps: int) -> Iterator[str]:
        yield mf([], [], "unsharp",
            luma_msize_x = self.radius,
            luma_msize_y = self.radius,
            luma_amount = self.strength,
            chroma_msize_x = self.radius,
            chroma_msize_y = self.radius,
            chroma_amount = self.strength,
        )


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


class TextOverlayFilter(VideoFilter):
    name = "Text overlay"

    text: str = Param("Text", value = "{frame}", ui_type = "box")
    anchor: FloatVector = Param("Anchor", axes = ["X", "Y"], minimum = 0.0, maximum = 1.0, step = 0.01, factory = lambda: FloatVector(0.0, 0.0), ui_type = "slider")
    offset: IntVector = Param("Offset", axes = ["X", "Y"], step = 1, factory = lambda: IntVector(0, 0), ui_type = "box")
    font: str = Param("Font", value = "sans", ui_type = "box")
    font_size: int = Param("Font size", minimum = 1, maximum = 144, step = 1, value = 16, ui_type = "slider")
    text_color: Color = Param("Text color", channels = 4, factory = lambda: Color(1.0, 1.0, 1.0, 1.0))
    shadow_offset: IntVector = Param("Shadow offset", axes = ["X", "Y"], step = 1, factory = lambda: IntVector(1, 1), ui_type = "box")
    shadow_color: Color = Param("Shadow color", channels = 4, factory = lambda: Color(0.0, 0.0, 0.0, 1.0))

    def generate(self, fps: int) -> Iterator[str]:
        yield mf([], [], "drawtext",
            text = self.text.format(frame = f"%{{eif:t*{fps}+1:d:5}}"),
            x = f"(W-tw)*{self.anchor.x}+{self.offset.x}",
            y = f"(H-th)*{self.anchor.y}+{self.offset.y}",
            font = self.font,
            fontsize = self.font_size,
            fontcolor = self.text_color.to_hex(),
            shadowx = self.shadow_offset.x,
            shadowy = self.shadow_offset.y,
            shadowcolor = self.shadow_color.to_hex(),
        )


def make_filter(inputs: list[str], outputs: list[str], name: str, *args: int | float | str, **kwargs: int | float | str) -> str:
    def sanitize(value: int | float | str) -> str:
        if isinstance(value, (int, float)):
            return str(value)
        else:
            return (
                value
                .replace("\\", "\\\\")
                .replace(":", "\\:")
                .replace("'", "\\'")
            )

    def generate_io(lst: list[str]) -> Iterator[str]:
        for slot in lst:
            yield f"[{slot}]"

    def generate_args() -> Iterator[str]:
        for value in args:
            yield sanitize(value)

        for key, value in kwargs.items():
            yield f"{key}={sanitize(value)}"

    return f"{''.join(generate_io(inputs))}{name}='{':'.join(generate_args())}'{''.join(generate_io(outputs))}"


mf = make_filter
