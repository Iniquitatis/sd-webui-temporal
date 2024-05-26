from abc import abstractmethod
from types import SimpleNamespace
from typing import Type

import gradio as gr

from temporal.meta.configurable import Configurable, UIParam
from temporal.utils.collection import reorder_dict


FILTERS: dict[str, Type["VideoFilter"]] = {}


def build_filter(ext_params: SimpleNamespace) -> str:
    return ",".join([
        filter.print(ext_params.video_fps, SimpleNamespace(**{
            x.id: getattr(ext_params, f"video_{id}_{x.id}")
            for x in filter.params.values()
        }))
        for id, filter in reorder_dict(FILTERS, ext_params.video_filtering_order or []).items()
        if getattr(ext_params, f"video_{id}_enabled")
    ] or ["null"])


class VideoFilter(Configurable, abstract = True):
    store = FILTERS

    @staticmethod
    @abstractmethod
    def print(fps: int, params: SimpleNamespace) -> str:
        raise NotImplementedError


class ChromaticAberrationFilter(VideoFilter):
    id = "chromatic_aberration"
    name = "Chromatic aberration"
    params = {
        "distance": UIParam("Distance", gr.Slider, minimum = 1, maximum = 512, step = 1, value = 1),
    }

    @staticmethod
    def print(fps: int, params: SimpleNamespace) -> str:
        return f"rgbashift='rh=-{params.distance}:bh={params.distance}'"


class ColorBalancingFilter(VideoFilter):
    id = "color_balancing"
    name = "Color balancing"
    params = {
        "brightness": UIParam("Brightness", gr.Slider, minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0),
        "contrast": UIParam("Contrast", gr.Slider, minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0),
        "saturation": UIParam("Saturation", gr.Slider, minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0),
    }

    @staticmethod
    def print(fps: int, params: SimpleNamespace) -> str:
        return f"eq='contrast={params.contrast}:brightness={params.brightness - 1.0}:saturation={params.saturation}'"


class DeflickeringFilter(VideoFilter):
    id = "deflickering"
    name = "Deflickering"
    params = {
        "frames": UIParam("Frames", gr.Slider, minimum = 2, maximum = 120, step = 1, value = 60),
    }

    @staticmethod
    def print(fps: int, params: SimpleNamespace) -> str:
        return f"deflicker='size={params.frames}:mode=am'"


class InterpolationFilter(VideoFilter):
    id = "interpolation"
    name = "Interpolation"
    params = {
        "fps": UIParam("Frames per second", gr.Slider, minimum = 1, maximum = 60, step = 1, value = 60),
        "mb_subframes": UIParam("Motion blur subframes", gr.Slider, minimum = 0, maximum = 15, step = 1, value = 0),
    }

    @staticmethod
    def print(fps: int, params: SimpleNamespace) -> str:
        parts = []
        parts.append(f"minterpolate='fps={params.fps * (params.mb_subframes + 1)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=none'")

        if params.mb_subframes > 0:
            parts.append(f"tmix='frames={params.mb_subframes + 1}'")
            parts.append(f"fps='{params.fps}'")

        return ",".join(parts)


class ScalingFilter(VideoFilter):
    id = "scaling"
    name = "Scaling"
    params = {
        "width": UIParam("Width", gr.Slider, minimum = 16, maximum = 2560, step = 8, value = 512),
        "height": UIParam("Height", gr.Slider, minimum = 16, maximum = 2560, step = 8, value = 512),
        "padded": UIParam("Padded", gr.Checkbox, value = False),
        "background_color": UIParam("Background color", gr.ColorPicker, value = "#000000"),
        "backdrop": UIParam("Backdrop", gr.Checkbox, value = False),
        "backdrop_brightness": UIParam("Backdrop brightness", gr.Slider, minimum = 0.0, maximum = 2.0, step = 0.01, value = 0.5),
        "backdrop_blurring": UIParam("Backdrop blurring", gr.Slider, minimum = 0.0, maximum = 50.0, step = 1.0, value = 0.0),
    }

    @staticmethod
    def print(fps: int, params: SimpleNamespace) -> str:
        parts = []

        if params.padded:
            parts.append(f"split[bg][fg]")

            if params.width > params.height:
                bgsw, bgsh = params.width, -1
                fgsw, fgsh = -1, params.height
                pw, ph = params.width, "ih"
                px, py = "(ow-iw)/2", 0
            else:
                bgsw, bgsh = -1, params.height
                fgsw, fgsh = params.width, -1
                pw, ph = "iw", params.height
                px, py = 0, "(oh-ih)/2"

            if params.backdrop:
                parts.append(f"[bg]scale='w={bgsw}:h={bgsh}:flags=lanczos'[bg]")
                parts.append(f"[bg]crop='w={params.width}:h={params.height}'[bg]")
                parts.append(f"[bg]eq='brightness={params.backdrop_brightness - 1.0}'[bg]")
                parts.append(f"[bg]gblur='sigma={params.backdrop_blurring}'[bg]")
            else:
                parts.append(f"[bg]scale='w={params.width}:h={params.height}:flags=neighbor'[bg]")
                parts.append(f"[bg]drawbox='w={params.width}:h={params.height}:color={params.background_color}:thickness=fill'[bg]")

            parts.append(f"[fg]scale='w={fgsw}:h={fgsh}:flags=lanczos'[fg]")
            parts.append(f"[fg]pad='w={pw}:h={ph}:x={px}:y={py}:color=#00000000'[fg]")
            parts.append(f"[bg][fg]overlay")
        else:
            parts.append(f"scale='{params.width}:{params.height}:flags=lanczos'")

        return ",".join(parts)


class SharpeningFilter(VideoFilter):
    id = "sharpening"
    name = "Sharpening"
    params = {
        "strength": UIParam("Strength", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.1, value = 0.0),
        "radius": UIParam("Radius", gr.Slider, minimum = 3, maximum = 13, step = 2, value = 3),
    }

    @staticmethod
    def print(fps: int, params: SimpleNamespace) -> str:
        return f"unsharp='luma_msize_x={params.radius}:luma_msize_y={params.radius}:luma_amount={params.strength}:chroma_msize_x={params.radius}:chroma_msize_y={params.radius}:chroma_amount={params.strength}'"


class TemporalAveragingFilter(VideoFilter):
    id = "temporal_averaging"
    name = "Temporal averaging"
    params = {
        "radius": UIParam("Radius", gr.Slider, minimum = 1, maximum = 60, step = 1, value = 1),
        "algorithm": UIParam("Algorithm", gr.Dropdown, choices = ["mean", "median"], value = "mean"),
        "easing": UIParam("Easing", gr.Slider, minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0),
    }

    @staticmethod
    def print(fps: int, params: SimpleNamespace) -> str:
        if params.algorithm == "mean":
            weights = [((x + 1) / (params.radius + 1)) ** params.easing for x in range(params.radius + 1)]
            weights += reversed(weights[:-1])
            weights = [f"{x:.18f}" for x in weights]
            return f"tmix='frames={len(weights)}:weights={' '.join(weights)}'"
        elif params.algorithm == "median":
            return f"tpad='start={params.radius}:stop={params.radius}:start_mode=clone:stop_mode=clone',tmedian='radius={params.radius}'"
        else:
            return "null"


class TextOverlayFilter(VideoFilter):
    id = "text_overlay"
    name = "Text overlay"
    params = {
        "text": UIParam("Text", gr.Textbox, value = "{frame}"),
        "anchor_x": UIParam("Anchor X", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0),
        "anchor_y": UIParam("Anchor Y", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0),
        "offset_x": UIParam("Offset X", gr.Number, precision = 1, step = 1, value = 0),
        "offset_y": UIParam("Offset Y", gr.Number, precision = 1, step = 1, value = 0),
        "font": UIParam("Font", gr.Textbox, value = "sans"),
        "font_size": UIParam("Font size", gr.Number, precision = 0, minimum = 1, maximum = 144, step = 1, value = 16),
        "text_color": UIParam("Text color", gr.ColorPicker, value = "#ffffff"),
        "text_alpha": UIParam("Text alpha", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0),
        "shadow_offset_x": UIParam("Shadow offset X", gr.Number, step = 1, value = 1),
        "shadow_offset_y": UIParam("Shadow offset Y", gr.Number, step = 1, value = 1),
        "shadow_color": UIParam("Shadow color", gr.ColorPicker, value = "#000000"),
        "shadow_alpha": UIParam("Shadow alpha", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0),
    }

    @staticmethod
    def print(fps: int, params: SimpleNamespace) -> str:
        text = (
            params.text
            .format(
                frame = f"%{{eif:t*{fps}+1:d:5}}",
            )
            .replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
        )
        return f"drawtext='text={text}:x=(W-tw)*{params.anchor_x}+{params.offset_x}:y=(H-th)*{params.anchor_y}+{params.offset_y}:font={params.font}:fontsize={params.font_size}:fontcolor={params.text_color}{int(params.text_alpha * 255.0):02x}:shadowx={params.shadow_offset_x}:shadowy={params.shadow_offset_y}:shadowcolor={params.shadow_color}{int(params.shadow_alpha * 255.0):02x}'"
