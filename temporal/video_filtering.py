from types import SimpleNamespace

import gradio as gr

from temporal.utils.collection import reorder_dict
from temporal.utils.func import make_func_registerer

FILTERS, filter = make_func_registerer(name = "", params = [])

def build_filter(ext_params):
    return ",".join([
        filter.func(ext_params.video_fps, SimpleNamespace(**{
            x.key: getattr(ext_params, f"video_{key}_{x.key}")
            for x in filter.params
        }))
        for key, filter in reorder_dict(FILTERS, ext_params.video_filtering_order or []).items()
        if getattr(ext_params, f"video_{key}_enabled")
    ] or ["null"])

class UIParam:
    def __init__(self, type, key, name, **kwargs):
        self.type = type
        self.key = key
        self.name = name
        self.kwargs = kwargs

@filter("chromatic_aberration", "Chromatic aberration", [
    UIParam(gr.Slider, "distance", "Distance", minimum = 1, maximum = 512, step = 1, value = 1),
])
def _(fps, params):
    return f"rgbashift='rh=-{params.distance}:bh={params.distance}'"

@filter("color_balancing", "Color balancing", [
    UIParam(gr.Slider, "brightness", "Brightness", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0),
    UIParam(gr.Slider, "contrast", "Contrast", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0),
    UIParam(gr.Slider, "saturation", "Saturation", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0),
])
def _(fps, params):
    return f"eq='contrast={params.contrast}:brightness={params.brightness - 1.0}:saturation={params.saturation}'"

@filter("deflickering", "Deflickering", [
    UIParam(gr.Slider, "frames", "Frames", minimum = 2, maximum = 120, step = 1, value = 60),
])
def _(fps, params):
    return f"deflicker='size={params.frames}:mode=am'"

@filter("interpolation", "Interpolation", [
    UIParam(gr.Slider, "fps", "Frames per second", minimum = 1, maximum = 60, step = 1, value = 60),
    UIParam(gr.Slider, "mb_subframes", "Motion blur subframes", minimum = 0, maximum = 15, step = 1, value = 0),
])
def _(fps, params):
    parts = []
    parts.append(f"minterpolate='fps={params.fps * (params.mb_subframes + 1)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=none'")

    if params.mb_subframes > 0:
        parts.append(f"tmix='frames={params.mb_subframes + 1}'")
        parts.append(f"fps='{params.fps}'")

    return ",".join(parts)

@filter("scaling", "Scaling", [
    UIParam(gr.Slider, "width", "Width", minimum = 16, maximum = 2560, step = 8, value = 512),
    UIParam(gr.Slider, "height", "Height", minimum = 16, maximum = 2560, step = 8, value = 512),
    UIParam(gr.Checkbox, "padded", "Padded", value = False),
    UIParam(gr.ColorPicker, "background_color", "Background color", value = "#000000"),
    UIParam(gr.Checkbox, "backdrop", "Backdrop", value = False),
    UIParam(gr.Slider, "backdrop_brightness", "Backdrop brightness", minimum = 0.0, maximum = 2.0, step = 0.01, value = 0.5),
    UIParam(gr.Slider, "backdrop_blurring", "Backdrop blurring", minimum = 0.0, maximum = 50.0, step = 1.0, value = 0.0),
])
def _(fps, params):
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

@filter("sharpening", "Sharpening", [
    UIParam(gr.Slider, "strength", "Strength", minimum = 0.0, maximum = 1.0, step = 0.1, value = 0.0),
    UIParam(gr.Slider, "radius", "Radius", minimum = 3, maximum = 13, step = 2, value = 3),
])
def _(fps, params):
    return f"unsharp='luma_msize_x={params.radius}:luma_msize_y={params.radius}:luma_amount={params.strength}:chroma_msize_x={params.radius}:chroma_msize_y={params.radius}:chroma_amount={params.strength}'"

@filter("temporal_averaging", "Temporal averaging", [
    UIParam(gr.Slider, "radius", "Radius", minimum = 1, maximum = 60, step = 1, value = 1),
    UIParam(gr.Dropdown, "algorithm", "Algorithm", choices = ["mean", "median"], value = "mean"),
    UIParam(gr.Slider, "easing", "Easing", minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0),
])
def _(fps, params):
    if params.algorithm == "mean":
        weights = [((x + 1) / (params.radius + 1)) ** params.easing for x in range(params.radius + 1)]
        weights += reversed(weights[:-1])
        weights = [f"{x:.18f}" for x in weights]
        return f"tmix='frames={len(weights)}:weights={' '.join(weights)}'"
    elif params.algorithm == "median":
        return f"tpad='start={params.radius}:stop={params.radius}:start_mode=clone:stop_mode=clone',tmedian='radius={params.radius}'"
    else:
        return "null"

@filter("text_overlay", "Text overlay", [
    UIParam(gr.Textbox, "text", "Text", value = "{frame}"),
    UIParam(gr.Slider, "anchor_x", "Anchor X", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0),
    UIParam(gr.Slider, "anchor_y", "Anchor Y", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0),
    UIParam(gr.Number, "offset_x", "Offset X", precision = 1, step = 1, value = 0),
    UIParam(gr.Number, "offset_y", "Offset Y", precision = 1, step = 1, value = 0),
    UIParam(gr.Textbox, "font", "Font", value = "sans"),
    UIParam(gr.Number, "font_size", "Font size", precision = 0, minimum = 1, maximum = 144, step = 1, value = 16),
    UIParam(gr.ColorPicker, "text_color", "Text color", value = "#ffffff"),
    UIParam(gr.Slider, "text_alpha", "Text alpha", minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0),
    UIParam(gr.Number, "shadow_offset_x", "Shadow offset X", step = 1, value = 1),
    UIParam(gr.Number, "shadow_offset_y", "Shadow offset Y", step = 1, value = 1),
    UIParam(gr.ColorPicker, "shadow_color", "Shadow color", value = "#000000"),
    UIParam(gr.Slider, "shadow_alpha", "Shadow alpha", minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0),
])
def _(fps, params):
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
