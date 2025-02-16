from abc import abstractmethod
from typing import Any, Type

from temporal.color import Color
from temporal.meta.configurable import BoolParam, ColorParam, Configurable, EnumParam, FloatParam, FloatVectorParam, IntParam, IntVectorParam, StringParam
from temporal.meta.serializable import SerializableField as Field
from temporal.utils.collection import find_by_predicate
from temporal.vector import FloatVector, IntVector


VIDEO_FILTERS: list[Type["VideoFilter"]] = []


class VideoFilter(Configurable, abstract = True):
    store = VIDEO_FILTERS

    enabled: bool = Field(True)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "VideoFilter":
        id = data.pop("id", "")

        if type := find_by_predicate(VIDEO_FILTERS, lambda x: x.id == id):
            return type.from_json(data)
        else:
            return super().from_json(data)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id} | super().to_json()

    @abstractmethod
    def print(self, fps: int) -> str:
        raise NotImplementedError


class ChromaticAberrationFilter(VideoFilter):
    name = "Chromatic aberration"

    distance: int = IntParam("Distance", minimum = 1, maximum = 512, step = 1, value = 1, ui_type = "slider")

    def print(self, fps: int) -> str:
        return f"rgbashift='rh=-{self.distance}:bh={self.distance}'"


class ColorBalancingFilter(VideoFilter):
    name = "Color balancing"

    brightness: float = FloatParam("Brightness", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, ui_type = "slider")
    contrast: float = FloatParam("Contrast", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, ui_type = "slider")
    saturation: float = FloatParam("Saturation", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, ui_type = "slider")

    def print(self, fps: int) -> str:
        return f"eq='contrast={self.contrast}:brightness={self.brightness - 1.0}:saturation={self.saturation}'"


class DebandingFilter(VideoFilter):
    name = "Debanding"

    radius: int = IntParam("Radius", minimum = 1, maximum = 64, step = 1, value = 16, ui_type = "slider")

    def print(self, fps: int) -> str:
        return f"deband='range={self.radius}'"


class DeflickeringFilter(VideoFilter):
    name = "Deflickering"

    frames: int = IntParam("Frames", minimum = 2, maximum = 120, step = 1, value = 60, ui_type = "slider")

    def print(self, fps: int) -> str:
        return f"deflicker='size={self.frames}:mode=am'"


class InterpolationFilter(VideoFilter):
    name = "Interpolation"

    fps: int = IntParam("Frames per second", minimum = 1, maximum = 60, step = 1, value = 60, ui_type = "slider")
    mb_subframes: int = IntParam("Motion blur subframes", minimum = 0, maximum = 15, step = 1, value = 0, ui_type = "slider")

    def print(self, fps: int) -> str:
        parts = []
        parts.append(f"minterpolate='fps={self.fps * (self.mb_subframes + 1)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=none'")

        if self.mb_subframes > 0:
            parts.append(f"tmix='frames={self.mb_subframes + 1}'")
            parts.append(f"fps='{self.fps}'")

        return ",".join(parts)


class ScalingFilter(VideoFilter):
    name = "Scaling"

    size: IntVector = IntVectorParam("Size", axes = ["Width", "Height"], minimum = 16, maximum = 2560, step = 8, factory = lambda: IntVector(512, 512), ui_type = "slider")
    padded: bool = BoolParam("Padded", value = False)
    background_color: Color = ColorParam("Background color", channels = 3, factory = lambda: Color(0.0, 0.0, 0.0))
    backdrop: bool = BoolParam("Backdrop", value = False)
    backdrop_brightness: float = FloatParam("Backdrop brightness", minimum = 0.0, maximum = 2.0, step = 0.01, value = 0.5, ui_type = "slider")
    backdrop_blurring: float = FloatParam("Backdrop blurring", minimum = 0.0, maximum = 50.0, step = 1.0, value = 0.0, ui_type = "slider")

    def print(self, fps: int) -> str:
        parts = []

        if self.padded:
            parts.append(f"split[bg][fg]")

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
                parts.append(f"[bg]scale='w={bgsw}:h={bgsh}:flags=lanczos'[bg]")
                parts.append(f"[bg]crop='w={self.size.x}:h={self.size.y}'[bg]")
                parts.append(f"[bg]eq='brightness={self.backdrop_brightness - 1.0}'[bg]")
                parts.append(f"[bg]gblur='sigma={self.backdrop_blurring}'[bg]")
            else:
                parts.append(f"[bg]scale='w={self.size.x}:h={self.size.y}:flags=neighbor'[bg]")
                parts.append(f"[bg]drawbox='w={self.size.x}:h={self.size.y}:color={self.background_color}:thickness=fill'[bg]")

            parts.append(f"[fg]scale='w={fgsw}:h={fgsh}:flags=lanczos'[fg]")
            parts.append(f"[fg]pad='w={pw}:h={ph}:x={px}:y={py}:color=#00000000'[fg]")
            parts.append(f"[bg][fg]overlay")
        else:
            parts.append(f"scale='{self.size.x}:{self.size.y}:flags=lanczos'")

        return ",".join(parts)


class SharpeningFilter(VideoFilter):
    name = "Sharpening"

    strength: float = FloatParam("Strength", minimum = 0.0, maximum = 1.0, step = 0.1, value = 0.0, ui_type = "slider")
    radius: int = IntParam("Radius", minimum = 3, maximum = 13, step = 2, value = 3, ui_type = "slider")

    def print(self, fps: int) -> str:
        return f"unsharp='luma_msize_x={self.radius}:luma_msize_y={self.radius}:luma_amount={self.strength}:chroma_msize_x={self.radius}:chroma_msize_y={self.radius}:chroma_amount={self.strength}'"


class TemporalAveragingFilter(VideoFilter):
    name = "Temporal averaging"

    radius: int = IntParam("Radius", minimum = 1, maximum = 60, step = 1, value = 1, ui_type = "slider")
    algorithm: str = EnumParam("Algorithm", choices = [("mean", "Mean"), ("median", "Median")], value = "mean", ui_type = "menu")
    easing: float = FloatParam("Easing", minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0, ui_type = "slider")

    def print(self, fps: int) -> str:
        if self.algorithm == "mean":
            weights = [((x + 1) / (self.radius + 1)) ** self.easing for x in range(self.radius + 1)]
            weights += reversed(weights[:-1])
            weights = [f"{x:.18f}" for x in weights]
            return f"tmix='frames={len(weights)}:weights={' '.join(weights)}'"
        elif self.algorithm == "median":
            return f"tpad='start={self.radius}:stop={self.radius}:start_mode=clone:stop_mode=clone',tmedian='radius={self.radius}'"
        else:
            return "null"


class TextOverlayFilter(VideoFilter):
    name = "Text overlay"

    text: str = StringParam("Text", value = "{frame}", ui_type = "box")
    anchor: FloatVector = FloatVectorParam("Anchor", axes = ["X", "Y"], minimum = 0.0, maximum = 1.0, step = 0.01, factory = lambda: FloatVector(0.0, 0.0), ui_type = "slider")
    offset: IntVector = IntVectorParam("Offset", axes = ["X", "Y"], step = 1, factory = lambda: IntVector(0, 0), ui_type = "box")
    font: str = StringParam("Font", value = "sans", ui_type = "box")
    font_size: int = IntParam("Font size", minimum = 1, maximum = 144, step = 1, value = 16, ui_type = "slider")
    text_color: Color = ColorParam("Text color", channels = 4, factory = lambda: Color(1.0, 1.0, 1.0, 1.0))
    shadow_offset: IntVector = IntVectorParam("Shadow offset", axes = ["X", "Y"], step = 1, factory = lambda: IntVector(1, 1), ui_type = "box")
    shadow_color: Color = ColorParam("Shadow color", channels = 4, factory = lambda: Color(0.0, 0.0, 0.0, 1.0))

    def print(self, fps: int) -> str:
        text = (
            self.text
            .format(
                frame = f"%{{eif:t*{fps}+1:d:5}}",
            )
            .replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
        )
        return f"drawtext='text={text}:x=(W-tw)*{self.anchor.x}+{self.offset.x}:y=(H-th)*{self.anchor.y}+{self.offset.y}:font={self.font}:fontsize={self.font_size}:fontcolor={self.text_color.to_hex()}:shadowx={self.shadow_offset.x}:shadowy={self.shadow_offset.y}:shadowcolor={self.shadow_color.to_hex()}'"
