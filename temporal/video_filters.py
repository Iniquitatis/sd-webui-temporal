from abc import abstractmethod
from typing import Type

import gradio as gr

from temporal.meta.configurable import Configurable, ui_param
from temporal.meta.serializable import field


VIDEO_FILTERS: dict[str, Type["VideoFilter"]] = {}


class VideoFilter(Configurable, abstract = True):
    store = VIDEO_FILTERS

    enabled: bool = field(False)

    @abstractmethod
    def print(self, fps: int) -> str:
        raise NotImplementedError


class ChromaticAberrationFilter(VideoFilter):
    id = "chromatic_aberration"
    name = "Chromatic aberration"

    distance: int = ui_param("Distance", gr.Slider, precision = 0, minimum = 1, maximum = 512, step = 1, value = 1)

    def print(self, fps: int) -> str:
        return f"rgbashift='rh=-{self.distance}:bh={self.distance}'"


class ColorBalancingFilter(VideoFilter):
    id = "color_balancing"
    name = "Color balancing"

    brightness: float = ui_param("Brightness", gr.Slider, minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0)
    contrast: float = ui_param("Contrast", gr.Slider, minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0)
    saturation: float = ui_param("Saturation", gr.Slider, minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0)

    def print(self, fps: int) -> str:
        return f"eq='contrast={self.contrast}:brightness={self.brightness - 1.0}:saturation={self.saturation}'"


class DeflickeringFilter(VideoFilter):
    id = "deflickering"
    name = "Deflickering"

    frames: int = ui_param("Frames", gr.Slider, precision = 0, minimum = 2, maximum = 120, step = 1, value = 60)

    def print(self, fps: int) -> str:
        return f"deflicker='size={self.frames}:mode=am'"


class InterpolationFilter(VideoFilter):
    id = "interpolation"
    name = "Interpolation"

    fps: int = ui_param("Frames per second", gr.Slider, precision = 0, minimum = 1, maximum = 60, step = 1, value = 60)
    mb_subframes: int = ui_param("Motion blur subframes", gr.Slider, precision = 0, minimum = 0, maximum = 15, step = 1, value = 0)

    def print(self, fps: int) -> str:
        parts = []
        parts.append(f"minterpolate='fps={self.fps * (self.mb_subframes + 1)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=none'")

        if self.mb_subframes > 0:
            parts.append(f"tmix='frames={self.mb_subframes + 1}'")
            parts.append(f"fps='{self.fps}'")

        return ",".join(parts)


class ScalingFilter(VideoFilter):
    id = "scaling"
    name = "Scaling"

    width: int = ui_param("Width", gr.Slider, precision = 0, minimum = 16, maximum = 2560, step = 8, value = 512)
    height: int = ui_param("Height", gr.Slider, precision = 0, minimum = 16, maximum = 2560, step = 8, value = 512)
    padded: bool = ui_param("Padded", gr.Checkbox, value = False)
    background_color: str = ui_param("Background color", gr.ColorPicker, value = "#000000")
    backdrop: bool = ui_param("Backdrop", gr.Checkbox, value = False)
    backdrop_brightness: float = ui_param("Backdrop brightness", gr.Slider, minimum = 0.0, maximum = 2.0, step = 0.01, value = 0.5)
    backdrop_blurring: float = ui_param("Backdrop blurring", gr.Slider, minimum = 0.0, maximum = 50.0, step = 1.0, value = 0.0)

    def print(self, fps: int) -> str:
        parts = []

        if self.padded:
            parts.append(f"split[bg][fg]")

            if self.width > self.height:
                bgsw, bgsh = self.width, -1
                fgsw, fgsh = -1, self.height
                pw, ph = self.width, "ih"
                px, py = "(ow-iw)/2", 0
            else:
                bgsw, bgsh = -1, self.height
                fgsw, fgsh = self.width, -1
                pw, ph = "iw", self.height
                px, py = 0, "(oh-ih)/2"

            if self.backdrop:
                parts.append(f"[bg]scale='w={bgsw}:h={bgsh}:flags=lanczos'[bg]")
                parts.append(f"[bg]crop='w={self.width}:h={self.height}'[bg]")
                parts.append(f"[bg]eq='brightness={self.backdrop_brightness - 1.0}'[bg]")
                parts.append(f"[bg]gblur='sigma={self.backdrop_blurring}'[bg]")
            else:
                parts.append(f"[bg]scale='w={self.width}:h={self.height}:flags=neighbor'[bg]")
                parts.append(f"[bg]drawbox='w={self.width}:h={self.height}:color={self.background_color}:thickness=fill'[bg]")

            parts.append(f"[fg]scale='w={fgsw}:h={fgsh}:flags=lanczos'[fg]")
            parts.append(f"[fg]pad='w={pw}:h={ph}:x={px}:y={py}:color=#00000000'[fg]")
            parts.append(f"[bg][fg]overlay")
        else:
            parts.append(f"scale='{self.width}:{self.height}:flags=lanczos'")

        return ",".join(parts)


class SharpeningFilter(VideoFilter):
    id = "sharpening"
    name = "Sharpening"

    strength: float = ui_param("Strength", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.1, value = 0.0)
    radius: int = ui_param("Radius", gr.Slider, precision = 0, minimum = 3, maximum = 13, step = 2, value = 3)

    def print(self, fps: int) -> str:
        return f"unsharp='luma_msize_x={self.radius}:luma_msize_y={self.radius}:luma_amount={self.strength}:chroma_msize_x={self.radius}:chroma_msize_y={self.radius}:chroma_amount={self.strength}'"


class TemporalAveragingFilter(VideoFilter):
    id = "temporal_averaging"
    name = "Temporal averaging"

    radius: int = ui_param("Radius", gr.Slider, precision = 0, minimum = 1, maximum = 60, step = 1, value = 1)
    algorithm: str = ui_param("Algorithm", gr.Dropdown, choices = ["mean", "median"], value = "mean")
    easing: float = ui_param("Easing", gr.Slider, minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0)

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
    id = "text_overlay"
    name = "Text overlay"

    text: str = ui_param("Text", gr.Textbox, value = "{frame}")
    anchor_x: float = ui_param("Anchor X", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0)
    anchor_y: float = ui_param("Anchor Y", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0)
    offset_x: int = ui_param("Offset X", gr.Number, precision = 0, step = 1, value = 0)
    offset_y: int = ui_param("Offset Y", gr.Number, precision = 0, step = 1, value = 0)
    font: str = ui_param("Font", gr.Textbox, value = "sans")
    font_size: int = ui_param("Font size", gr.Number, precision = 0, minimum = 1, maximum = 144, step = 1, value = 16)
    text_color: str = ui_param("Text color", gr.ColorPicker, value = "#ffffff")
    text_alpha: float = ui_param("Text alpha", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0)
    shadow_offset_x: int = ui_param("Shadow offset X", gr.Number, precision = 0, step = 1, value = 1)
    shadow_offset_y: int = ui_param("Shadow offset Y", gr.Number, precision = 0, step = 1, value = 1)
    shadow_color: str = ui_param("Shadow color", gr.ColorPicker, value = "#000000")
    shadow_alpha: float = ui_param("Shadow alpha", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0)

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
        return f"drawtext='text={text}:x=(W-tw)*{self.anchor_x}+{self.offset_x}:y=(H-th)*{self.anchor_y}+{self.offset_y}:font={self.font}:fontsize={self.font_size}:fontcolor={self.text_color}{int(self.text_alpha * 255.0):02x}:shadowx={self.shadow_offset_x}:shadowy={self.shadow_offset_y}:shadowcolor={self.shadow_color}{int(self.shadow_alpha * 255.0):02x}'"
