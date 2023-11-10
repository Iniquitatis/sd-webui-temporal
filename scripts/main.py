import json
from contextlib import contextmanager
from copy import copy
from io import BytesIO
from itertools import count
from pathlib import Path
from shutil import rmtree
from subprocess import run
from threading import Lock, Thread
from time import sleep
from types import SimpleNamespace

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
from PIL import Image, ImageColor

from modules import images, processing, scripts
from modules.shared import opts, prompt_styles, state

#===============================================================================

EXTENSION_DIR = Path(scripts.basedir())

def import_cn():
    try:
        from scripts import external_code
    except:
        external_code = None

    return external_code

#===============================================================================

def safe_get_directory(path):
    path.mkdir(parents = True, exist_ok = True)
    return path

#===============================================================================

class ThreadQueue:
    def __init__(self):
        self._queue = []
        self._execution_lock = Lock()
        self._queue_lock = Lock()

    @property
    def busy(self):
        with self._queue_lock:
            return len(self._queue) > 0

    def enqueue(self, target, *args, **kwargs):
        def callback():
            with self._execution_lock:
                target(*args, **kwargs)

            with self._queue_lock:
                self._queue.pop(0)

        with self._queue_lock:
            thread = Thread(target = callback)
            self._queue.append(thread)
            thread.start()

#===============================================================================

def lerp(a, b, x):
    return a * (1.0 - x) + b * x

def remap_range(value, old_min, old_max, new_min, new_max):
    return new_min + (value - old_min) / (old_max - old_min) * (new_max - new_min)

#===============================================================================

BLEND_MODES = dict()

def blend_mode(key, name):
    def decorator(func):
        BLEND_MODES[key] = dict(name = name, func = func)
        return func
    return decorator

@blend_mode("normal", "Normal")
def _(b, s):
    return s

@blend_mode("add", "Add")
def _(b, s):
    return b + s

@blend_mode("subtract", "Subtract")
def _(b, s):
    return b - s

@blend_mode("multiply", "Multiply")
def _(b, s):
    return b * s

@blend_mode("divide", "Divide")
def _(b, s):
    return b / np.clip(s, 1e-6, 1.0)

@blend_mode("lighten", "Lighten")
def _(b, s):
    return np.maximum(b, s)

@blend_mode("darken", "Darken")
def _(b, s):
    return np.minimum(b, s)

@blend_mode("hard_light", "Hard light")
def _(b, s):
    result = np.zeros_like(s)
    less_idx = np.where(s <= 0.5)
    more_idx = np.where(s >  0.5)
    result[less_idx] = BLEND_MODES["multiply"]["func"](b[less_idx], 2.0 * s[less_idx])
    result[more_idx] = BLEND_MODES["screen"]["func"](b[more_idx], 2.0 * s[more_idx] - 1.0)
    return result

@blend_mode("soft_light", "Soft light")
def _(b, s):
    def D(b):
        result = np.zeros_like(b)
        less_idx = np.where(b <= 0.25)
        more_idx = np.where(b >  0.25)
        result[less_idx] = ((16.0 * b[less_idx] - 12.0) * b[less_idx] + 4.0) * b[less_idx]
        result[more_idx] = np.sqrt(b[more_idx])
        return result

    result = np.zeros_like(s)
    less_idx = np.where(s <= 0.5)
    more_idx = np.where(s >  0.5)
    result[less_idx] = b[less_idx] - (1.0 - 2.0 * s[less_idx]) * b[less_idx] * (1.0 - b[less_idx])
    result[more_idx] = b[more_idx] + (2.0 * s[more_idx] - 1.0) * (D(b[more_idx]) - b[more_idx])
    return result

@blend_mode("color_dodge", "Color dodge")
def _(b, s):
    result = np.zeros_like(s)
    b0_mask = b == 0.0
    s1_mask = s == 1.0
    else_mask = np.logical_not(np.logical_or(b0_mask, s1_mask))
    else_idx = np.where(else_mask)
    result[np.where(b0_mask)] = 0.0
    result[np.where(s1_mask)] = 1.0
    result[else_idx] = np.minimum(1.0, b[else_idx] / (1.0 - s[else_idx]))
    return result

@blend_mode("color_burn", "Color burn")
def _(b, s):
    result = np.zeros_like(s)
    b1_mask = b == 1.0
    s0_mask = s == 0.0
    else_mask = np.logical_not(np.logical_or(b1_mask, s0_mask))
    else_idx = np.where(else_mask)
    result[np.where(b1_mask)] = 1.0
    result[np.where(s0_mask)] = 0.0
    result[else_idx] = 1.0 - np.minimum(1.0, (1.0 - b[else_idx]) / s[else_idx])
    return result

@blend_mode("overlay", "Overlay")
def _(b, s):
    return BLEND_MODES["hard_light"]["func"](s, b)

@blend_mode("screen", "Screen")
def _(b, s):
    return b + s - (b * s)

@blend_mode("difference", "Difference")
def _(b, s):
    return np.abs(b - s)

@blend_mode("exclusion", "Exclusion")
def _(b, s):
    return b + s - 2.0 * b * s

@blend_mode("hue", "Hue")
def _(b, s):
    b_hsv = skimage.color.rgb2hsv(b)
    s_hsv = skimage.color.rgb2hsv(s)
    return skimage.color.hsv2rgb(np.stack((s_hsv[..., 0], b_hsv[..., 1], b_hsv[..., 2]), axis = 2))

@blend_mode("saturation", "Saturation")
def _(b, s):
    b_hsv = skimage.color.rgb2hsv(b)
    s_hsv = skimage.color.rgb2hsv(s)
    return skimage.color.hsv2rgb(np.stack((b_hsv[..., 0], s_hsv[..., 1], b_hsv[..., 2]), axis = 2))

@blend_mode("value", "Value")
def _(b, s):
    b_hsv = skimage.color.rgb2hsv(b)
    s_hsv = skimage.color.rgb2hsv(s)
    return skimage.color.hsv2rgb(np.stack((b_hsv[..., 0], b_hsv[..., 1], s_hsv[..., 2]), axis = 2))

@blend_mode("color", "Color")
def _(b, s):
    b_hsv = skimage.color.rgb2hsv(b)
    s_hsv = skimage.color.rgb2hsv(s)
    return skimage.color.hsv2rgb(np.stack((s_hsv[..., 0], s_hsv[..., 1], b_hsv[..., 2]), axis = 2))

def blend_images(npim, modulator, mode, amount):
    if modulator is None or amount <= 0.0:
        return npim

    return lerp(npim, BLEND_MODES[mode]["func"](npim, modulator), amount)

def generate_noise_image(size, seed):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, size = (size[1], size[0], 3), dtype = "uint8"))

def load_image(path):
    im = Image.open(path)
    im.load()
    return im

def match_image(im, reference, mode = True, size = True):
    if mode and im.mode != reference.mode:
        im = im.convert(reference.mode)

    if size and im.size != reference.size:
        im = im.resize(reference.size, Image.Resampling.LANCZOS)

    return im

def preprocess_image(im, uv, seed):
    im = im.convert("RGB")
    npim = skimage.img_as_float(im)
    height, width = npim.shape[:2]

    if uv.noise_compression_enabled:
        weight = 0.0

        if uv.noise_compression_constant > 0.0:
            weight += uv.noise_compression_constant

        if uv.noise_compression_adaptive > 0.0:
            weight += skimage.restoration.estimate_sigma(npim, average_sigmas = True, channel_axis = 2) * uv.noise_compression_adaptive

        npim = skimage.restoration.denoise_tv_chambolle(npim, weight = max(weight, 1e-5), channel_axis = 2)

    if uv.color_correction_enabled:
        if uv.color_correction_image is not None:
            npim = skimage.exposure.match_histograms(npim, skimage.img_as_float(match_image(uv.color_correction_image, im, size = False)), channel_axis = 2)

        if uv.normalize_contrast:
            npim = skimage.exposure.rescale_intensity(npim)

    if uv.color_balancing_enabled:
        npim = remap_range(npim, npim.min(), npim.max(), 0.0, uv.brightness)

        npim = remap_range(npim, npim.min(), npim.max(), 0.5 - uv.contrast / 2, 0.5 + uv.contrast / 2)

        hsv = skimage.color.rgb2hsv(npim, channel_axis = 2)
        s = hsv[..., 1]
        s[:] = remap_range(s, s.min(), s.max(), s.min(), uv.saturation)
        npim = skimage.color.hsv2rgb(hsv)

    if uv.noise_enabled:
        npim = blend_images(npim, np.random.default_rng(seed).uniform(high = 1.0 + np.finfo(npim.dtype).eps, size = npim.shape), uv.noise_mode, uv.noise_amount)

    if uv.modulation_enabled and uv.modulation_image is not None:
        npim = blend_images(npim, skimage.filters.gaussian(skimage.img_as_float(match_image(uv.modulation_image, im)), uv.modulation_blurring, channel_axis = 2), uv.modulation_mode, uv.modulation_amount)

    if uv.tinting_enabled:
        npim = blend_images(npim, np.full_like(npim, np.array(ImageColor.getrgb(uv.tinting_color)) / 255.0), uv.tinting_mode, uv.tinting_amount)

    if uv.sharpening_enabled:
        npim = skimage.filters.unsharp_mask(npim, uv.sharpening_radius, uv.sharpening_amount, channel_axis = 2)

    if uv.transformation_enabled:
        o_transform = skimage.transform.AffineTransform(translation = (-width / 2, -height / 2))
        t_transform = skimage.transform.AffineTransform(translation = (-uv.translation_x * width, -uv.translation_y * height))
        r_transform = skimage.transform.AffineTransform(rotation = np.deg2rad(uv.rotation))
        s_transform = skimage.transform.AffineTransform(scale = uv.scaling)
        npim = skimage.transform.warp(npim, skimage.transform.AffineTransform(t_transform.params @ np.linalg.inv(o_transform.params) @ s_transform.params @ r_transform.params @ o_transform.params).inverse, mode = "symmetric")

    if uv.symmetrize:
        npim[:, width // 2:] = np.flip(npim[:, :width // 2], axis = 1)

    if uv.blurring_enabled:
        npim = skimage.filters.gaussian(npim, uv.blurring_radius, channel_axis = 2)

    if uv.custom_code_enabled:
        code_globals = dict(
            np = np,
            scipy = scipy,
            skimage = skimage,
            input = npim,
        )
        exec(uv.custom_code, code_globals)
        npim = code_globals.get("output", npim)

    return Image.fromarray(skimage.img_as_ubyte(np.clip(npim, 0.0, 1.0)))

#===============================================================================

def render_video(uv, is_final):
    output_dir = Path(uv.output_dir)
    frame_dir = output_dir / uv.project_subdir
    frame_paths = sorted(frame_dir.glob("*.png"), key = lambda x: x.name)
    video_path = output_dir / f"{uv.project_subdir}-{'final' if is_final else 'draft'}.mp4"

    if uv.video_looping:
        frame_paths += reversed(frame_paths[:-1])

    filters = []

    if is_final:
        if uv.video_deflickering_enabled:
            filters.append(f"deflicker='size={min(uv.video_deflickering_frames, len(frame_paths))}:mode=am'")

        if uv.video_interpolation_enabled:
            filters.append(f"minterpolate='fps={uv.video_interpolation_fps * (uv.video_interpolation_mb_subframes + 1)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=none'")

            if uv.video_interpolation_mb_subframes > 0:
                filters.append(f"tmix='frames={uv.video_interpolation_mb_subframes + 1}'")
                filters.append(f"fps='{uv.video_interpolation_fps}'")

        if uv.video_temporal_blurring_enabled:
            weights = [((x + 1) / (uv.video_temporal_blurring_radius + 1)) ** uv.video_temporal_blurring_easing for x in range(uv.video_temporal_blurring_radius + 1)]
            weights += reversed(weights[:-1])
            weights = [f"{x:.18f}" for x in weights]
            filters.append(f"tmix='frames={len(weights)}:weights={' '.join(weights)}'")

        if uv.video_scaling_enabled:
            filters.append(f"scale='{uv.video_scaling_width}x{uv.video_scaling_height}:flags=lanczos'")

    if uv.video_frame_num_overlay_enabled:
        filters.append(f"drawtext='text=%{{eif\\:n*{uv.video_fps / uv.video_interpolation_fps if is_final and uv.video_interpolation_enabled else 1.0:.18f}+1\\:d\\:5}}:x=5:y=5:fontsize={uv.video_frame_num_overlay_font_size}:fontcolor={uv.video_frame_num_overlay_text_color}{int(uv.video_frame_num_overlay_text_alpha * 255.0):02x}:shadowx=1:shadowy=1:shadowcolor={uv.video_frame_num_overlay_shadow_color}{int(uv.video_frame_num_overlay_shadow_alpha * 255.0):02x}'")

    run([
        "ffmpeg",
        "-y",
        "-r", str(uv.video_fps),
        "-f", "concat",
        "-protocol_whitelist", "fd,file",
        "-safe", "0",
        "-i", "-",
        "-framerate", str(uv.video_fps),
        "-vf", ",".join(filters) if len(filters) > 0 else "null",
        "-c:v", "libx264",
        "-crf", "14",
        "-preset", "slow" if is_final else "veryfast",
        "-tune", "film",
        "-pix_fmt", "yuv420p",
        video_path,
    ], input = "".join(f"file '{frame_path.resolve()}'\nduration 1\n" for frame_path in frame_paths).encode("utf-8"))

#===============================================================================

def get_last_frame_index(frame_dir):
    def get_index(path):
        try:
            if path.is_file():
                return int(path.stem)
        except:
            print(f"WARNING: {path} doesn't match the frame name format")
            return 0

    return max((get_index(path) for path in frame_dir.glob("*.png")), default = 0)

def load_object(obj, data, data_dir):
    def load_value(value):
        if isinstance(value, bool | int | float | str | None):
            return value

        elif isinstance(value, list):
            return [load_value(x) for x in value]

        elif isinstance(value, dict):
            im_type = value.get("im_type", "")
            im_path = data_dir / value.get("filename", "")

            if im_type == "pil":
                return load_image(im_path)
            elif im_type == "np":
                return np.array(load_image(im_path))
            else:
                return {k: load_value(v) for k, v in value.items()}

    for key, value in data.items():
        if hasattr(obj, key):
            setattr(obj, key, load_value(value))

def save_object(obj, data_dir, filter = None):
    def save_value(value):
        if isinstance(value, bool | int | float | str | None):
            return value

        elif isinstance(value, tuple):
            return tuple(save_value(x) for x in value)

        elif isinstance(value, list):
            return [save_value(x) for x in value]

        elif isinstance(value, dict):
            return {k: save_value(v) for k, v in value.items()}

        elif isinstance(value, Image.Image):
            filename = f"{id(value)}.png"
            value.save(data_dir / filename)
            return {"im_type": "pil", "filename": filename}

        elif isinstance(value, np.ndarray):
            filename = f"{id(value)}.png"
            Image.fromarray(value).save(data_dir / filename)
            return {"im_type": "np", "filename": filename}

    return {k: save_value(v) for k, v in vars(obj).items() if not filter or k in filter}

def load_session(p, uv, project_dir, session_dir, last_index):
    if not (params_path := (session_dir / "parameters.json")).is_file():
        return

    with open(params_path, "r", encoding = "utf-8") as params_file:
        data = json.load(params_file)

    load_object(p, data.get("generation_params", {}), session_dir)

    if external_code := import_cn():
        for unit_data, cn_unit in zip(data.get("controlnet_params", []), external_code.get_all_units_in_processing(p)):
            load_object(cn_unit, unit_data, session_dir)

    load_object(uv, data.get("extension_params", {}), session_dir)

    if (im_path := (project_dir / f"{last_index:05d}.png")).is_file():
        p.init_images = [load_image(im_path)]

    # FIXME: Unreliable; works properly only on first generation, but not on the subsequent ones
    if p.seed != -1:
        p.seed = p.seed + last_index

def save_session(p, uv, project_dir, session_dir, last_index):
    for path in session_dir.glob("*.*"):
        path.unlink()

    data = dict(
        generation_params = save_object(p, session_dir, [
            "prompt",
            "negative_prompt",
            "init_images",
            "resize_mode",
            "sampler_name",
            "steps",
            "width",
            "height",
            "cfg_scale",
            "denoising_strength",
            "seed",
            "seed_resize_from_w",
            "seed_resize_from_h",
        ]),
        controlnet_params = list(
            save_object(cn_unit, session_dir, [
                "enabled",
                "module",
                "model",
                "weight",
                "image",
                "resize_mode",
                "low_vram",
                "processor_res",
                "threshold_a",
                "threshold_b",
                "guidance_start",
                "guidance_end",
                "pixel_perfect",
                "control_mode",
            ])
            for cn_unit in external_code.get_all_units_in_processing(p)
        ) if (external_code := import_cn()) else [],
        extension_params = save_object(uv, session_dir, [
            "save_every_nth_frame",
            "noise_compression_enabled",
            "noise_compression_constant",
            "noise_compression_adaptive",
            "color_correction_enabled",
            "color_correction_image",
            "normalize_contrast",
            "color_balancing_enabled",
            "brightness",
            "contrast",
            "saturation",
            "noise_enabled",
            "noise_amount",
            "noise_relative",
            "noise_mode",
            "modulation_enabled",
            "modulation_amount",
            "modulation_relative",
            "modulation_mode",
            "modulation_image",
            "modulation_blurring",
            "tinting_enabled",
            "tinting_amount",
            "tinting_relative",
            "tinting_mode",
            "tinting_color",
            "sharpening_enabled",
            "sharpening_amount",
            "sharpening_relative",
            "sharpening_radius",
            "transformation_enabled",
            "scaling",
            "rotation",
            "translation_x",
            "translation_y",
            "symmetrize",
            "blurring_enabled",
            "blurring_radius",
            "custom_code_enabled",
            "custom_code",
        ]),
    )

    with open(session_dir / "parameters.json", "w", encoding = "utf-8") as params_file:
        json.dump(data, params_file, indent = 4)

def generate_image(job_title, p, **p_overrides):
    state.job = job_title

    p_instance = copy(p)

    for key, value in p_overrides.items():
        if hasattr(p_instance, key):
            setattr(p_instance, key, value)
        else:
            print(f"WARNING: Key {key} doesn't exist in {p_instance.__class__.__name__}")

    processed = processing.process_images(p_instance)

    if state.interrupted or state.skipped:
        return None

    return processed

#===============================================================================

class Metrics:
    def __init__(self):
        self.luminance_mean = []
        self.luminance_std = []
        self.color_level_mean = []
        self.color_level_std = []
        self.noise_sigma = []

    def measure(self, im):
        npim = skimage.img_as_float(im)
        grayscale = skimage.color.rgb2gray(npim[..., :3], channel_axis = 2)
        red, green, blue = npim[..., 0], npim[..., 1], npim[..., 2]

        self.luminance_mean.append(np.mean(grayscale))
        self.luminance_std.append(np.std(grayscale))
        self.color_level_mean.append([np.mean(red), np.mean(green), np.mean(blue)])
        self.color_level_std.append([np.std(red), np.std(green), np.std(blue)])
        self.noise_sigma.append(skimage.restoration.estimate_sigma(npim, average_sigmas = True, channel_axis = 2))

    def load(self, project_dir):
        metrics_dir = project_dir / "metrics"

        if (data_path := (metrics_dir / "data.json")).is_file():
            with open(data_path, "r", encoding = "utf-8") as data_file:
                load_object(self, json.load(data_file), metrics_dir)

    def save(self, project_dir):
        metrics_dir = safe_get_directory(project_dir / "metrics")

        with open(metrics_dir / "data.json", "w", encoding = "utf-8") as data_file:
            json.dump(save_object(self, metrics_dir), data_file, indent = 4)

    def clear(self, project_dir):
        self.luminance_mean.clear()
        self.luminance_std.clear()
        self.color_level_mean.clear()
        self.color_level_std.clear()
        self.noise_sigma.clear()

        if (metrics_dir := (project_dir / "metrics")).is_dir():
            rmtree(metrics_dir)

    def plot(self, project_dir, save_images = False):
        metrics_dir = safe_get_directory(project_dir / "metrics")

        result = []

        @contextmanager
        def figure(title, path):
            plt.title(title)
            plt.xlabel("Frame")
            plt.ylabel("Level")
            plt.grid()

            try:
                yield
            finally:
                plt.legend()

                buffer = BytesIO()
                plt.savefig(buffer, format = "png")
                buffer.seek(0)

                im = Image.open(buffer)
                im.load()

                if save_images:
                    im.save(path)

                result.append(im)

                plt.close()

        def plot_noise_graph(data, label, color):
            plt.axhline(data[0], color = color, linestyle = ":", linewidth = 0.5)
            plt.axhline(np.mean(data), color = color, linestyle = "--", linewidth = 1.0)
            plt.plot(data, color = color, label = label, linestyle = "--", linewidth = 0.5, marker = "+", markersize = 3)

            if data.size > 3:
                plt.plot(scipy.signal.savgol_filter(data, min(data.size, 51), 3), color = color, label = f"{label} (smoothed)", linestyle = "-")

        with figure("Luminance mean", metrics_dir / "luminance_mean.png"):
            plot_noise_graph(np.array(self.luminance_mean), "Luminance", "gray")

        with figure("Luminance standard deviation", metrics_dir / "luminance_std.png"):
            plot_noise_graph(np.array(self.luminance_std), "Luminance", "gray")

        with figure("Color level mean", metrics_dir / "color_level_mean.png"):
            plot_noise_graph(np.array(self.color_level_mean)[..., 0], "Red", "darkred")
            plot_noise_graph(np.array(self.color_level_mean)[..., 1], "Green", "darkgreen")
            plot_noise_graph(np.array(self.color_level_mean)[..., 2], "Blue", "darkblue")

        with figure("Color level standard deviation", metrics_dir / "color_level_std.png"):
            plot_noise_graph(np.array(self.color_level_std)[..., 0], "Red", "darkred")
            plot_noise_graph(np.array(self.color_level_std)[..., 1], "Green", "darkgreen")
            plot_noise_graph(np.array(self.color_level_std)[..., 2], "Blue", "darkblue")

        with figure("Noise sigma", metrics_dir / "noise_sigma.png"):
            plot_noise_graph(np.array(self.noise_sigma), "Noise sigma", "royalblue")

        return result

#===============================================================================

class TemporalScript(scripts.Script):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._image_save_tq = ThreadQueue()
        self._video_render_tq = ThreadQueue()

    def title(self):
        return "Temporal"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        ue = SimpleNamespace()
        ue_dict = vars(ue)

        with gr.Tab("General"):
            ue.output_dir = gr.Textbox(label = "Output directory", value = "outputs/temporal", elem_id = self.elem_id("output_dir"))
            ue.project_subdir = gr.Textbox(label = "Project subdirectory", value = "untitled", elem_id = self.elem_id("project_subdir"))
            ue.frame_count = gr.Number(label = "Frame count", precision = 0, minimum = 1, step = 1, value = 100, elem_id = self.elem_id("frame_count"))
            ue.save_every_nth_frame = gr.Number(label = "Save every N-th frame", precision = 0, minimum = 1, step = 1, value = 1, elem_id = self.elem_id("save_every_nth_frame"))
            ue.archive_mode = gr.Checkbox(label = "Archive mode", value = False, elem_id = self.elem_id("archive_mode"))
            ue.start_from_scratch = gr.Checkbox(label = "Start from scratch", value = False, elem_id = self.elem_id("start_from_scratch"))
            ue.load_session = gr.Checkbox(label = "Load session", value = True, elem_id = self.elem_id("load_session"))
            ue.save_session = gr.Checkbox(label = "Save session", value = True, elem_id = self.elem_id("save_session"))

        with gr.Tab("Frame Preprocessing"):
            with gr.Accordion("Noise compression"):
                ue.noise_compression_enabled = gr.Checkbox(label = "Enabled", value = False, elem_id = self.elem_id("noise_compression_enabled"))
                ue.noise_compression_constant = gr.Slider(label = "Constant", minimum = 0.0, maximum = 1.0, step = 1e-5, value = 0.0, elem_id = self.elem_id("noise_compression_constant"))
                ue.noise_compression_adaptive = gr.Slider(label = "Adaptive", minimum = 0.0, maximum = 2.0, step = 0.01, value = 0.0, elem_id = self.elem_id("noise_compression_adaptive"))

            with gr.Accordion("Color correction"):
                ue.color_correction_enabled = gr.Checkbox(label = "Enabled", value = False, elem_id = self.elem_id("color_correction_enabled"))
                ue.color_correction_image = gr.Pil(label = "Reference image", elem_id = self.elem_id("color_correction_image"))
                ue.normalize_contrast = gr.Checkbox(label = "Normalize contrast", value = False, elem_id = self.elem_id("normalize_contrast"))

            with gr.Accordion("Color balancing"):
                ue.color_balancing_enabled = gr.Checkbox(label = "Enabled", value = False, elem_id = self.elem_id("color_balancing_enabled"))
                ue.brightness = gr.Slider(label = "Brightness", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, elem_id = self.elem_id("brightness"))
                ue.contrast = gr.Slider(label = "Contrast", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, elem_id = self.elem_id("contrast"))
                ue.saturation = gr.Slider(label = "Saturation", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, elem_id = self.elem_id("saturation"))

            with gr.Accordion("Noise"):
                ue.noise_enabled = gr.Checkbox(label = "Enabled", value = False, elem_id = self.elem_id("noise_enabled"))

                with gr.Row():
                    ue.noise_amount = gr.Slider(label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, elem_id = self.elem_id("noise_amount"))
                    ue.noise_relative = gr.Checkbox(label = "Relative", value = False, elem_id = self.elem_id("noise_relative"))

                # FIXME: Pairs (name, value) don't work in older versions of Gradio
                ue.noise_mode = gr.Dropdown(label = "Mode", type = "value", choices = list(BLEND_MODES.keys()), value = next(iter(BLEND_MODES)), elem_id = self.elem_id("noise_mode"))

            with gr.Accordion("Modulation"):
                ue.modulation_enabled = gr.Checkbox(label = "Enabled", value = False, elem_id = self.elem_id("modulation_enabled"))

                with gr.Row():
                    ue.modulation_amount = gr.Slider(label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, elem_id = self.elem_id("modulation_amount"))
                    ue.modulation_relative = gr.Checkbox(label = "Relative", value = False, elem_id = self.elem_id("modulation_relative"))

                # FIXME: Pairs (name, value) don't work in older versions of Gradio
                ue.modulation_mode = gr.Dropdown(label = "Mode", type = "value", choices = list(BLEND_MODES.keys()), value = next(iter(BLEND_MODES)), elem_id = self.elem_id("modulation_mode"))
                ue.modulation_image = gr.Pil(label = "Image", elem_id = self.elem_id("modulation_image"))
                ue.modulation_blurring = gr.Slider(label = "Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0, elem_id = self.elem_id("modulation_blurring"))

            with gr.Accordion("Tinting"):
                ue.tinting_enabled = gr.Checkbox(label = "Enabled", value = False, elem_id = self.elem_id("tinting_enabled"))

                with gr.Row():
                    ue.tinting_amount = gr.Slider(label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, elem_id = self.elem_id("tinting_amount"))
                    ue.tinting_relative = gr.Checkbox(label = "Relative", value = False, elem_id = self.elem_id("tinting_relative"))

                # FIXME: Pairs (name, value) don't work in older versions of Gradio
                ue.tinting_mode = gr.Dropdown(label = "Mode", type = "value", choices = list(BLEND_MODES.keys()), value = next(iter(BLEND_MODES)), elem_id = self.elem_id("tinting_mode"))
                ue.tinting_color = gr.ColorPicker(label = "Color", value = "#ffffff", elem_id = self.elem_id("tinting_color"))

            with gr.Accordion("Sharpening"):
                ue.sharpening_enabled = gr.Checkbox(label = "Enabled", value = False, elem_id = self.elem_id("sharpening_enabled"))

                with gr.Row():
                    ue.sharpening_amount = gr.Slider(label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, elem_id = self.elem_id("sharpening_amount"))
                    ue.sharpening_relative = gr.Checkbox(label = "Relative", value = False, elem_id = self.elem_id("sharpening_relative"))

                ue.sharpening_radius = gr.Slider(label = "Radius", minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0, elem_id = self.elem_id("sharpening_radius"))

            with gr.Accordion("Transformation"):
                ue.transformation_enabled = gr.Checkbox(label = "Enabled", value = False, elem_id = self.elem_id("transformation_enabled"))

                with gr.Row():
                    ue.translation_x = gr.Number(label = "Translation X", step = 0.001, value = 0.0, elem_id = self.elem_id("translation_x"))
                    ue.translation_y = gr.Number(label = "Translation Y", step = 0.001, value = 0.0, elem_id = self.elem_id("translation_y"))

                ue.rotation = gr.Slider(label = "Rotation", minimum = -90.0, maximum = 90.0, step = 0.1, value = 0.0, elem_id = self.elem_id("rotation"))
                ue.scaling = gr.Slider(label = "Scaling", minimum = 0.0, maximum = 2.0, step = 0.001, value = 1.0, elem_id = self.elem_id("scaling"))

            ue.symmetrize = gr.Checkbox(label = "Symmetrize", value = False, elem_id = self.elem_id("symmetrize"))

            with gr.Accordion("Blurring"):
                ue.blurring_enabled = gr.Checkbox(label = "Enabled", value = False, elem_id = self.elem_id("blurring_enabled"))
                ue.blurring_radius = gr.Slider(label = "Radius", minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0, elem_id = self.elem_id("blurring_radius"))

            with gr.Accordion("Custom code"):
                ue.custom_code_enabled = gr.Checkbox(label = "Enabled", value = False, elem_id = self.elem_id("custom_code_enabled"))
                gr.Markdown("**WARNING:** Don't put an untrusted code here!")
                ue.custom_code = gr.Code(label = "Code", language = "python", value = "", elem_id = self.elem_id("custom_code"))

        with gr.Tab("Video Rendering"):
            ue.video_fps = gr.Slider(label = "Frames per second", minimum = 1, maximum = 60, step = 1, value = 30, elem_id = self.elem_id("video_fps"))
            ue.video_looping = gr.Checkbox(label = "Looping", value = False, elem_id = self.elem_id("video_looping"))

            with gr.Accordion("Deflickering"):
                ue.video_deflickering_enabled = gr.Checkbox(label = "Enabled", value = False, elem_id = self.elem_id("video_deflickering_enabled"))
                ue.video_deflickering_frames = gr.Slider(label = "Frames", minimum = 2, maximum = 120, step = 1, value = 60, elem_id = self.elem_id("video_deflickering_frames"))

            with gr.Accordion("Interpolation"):
                ue.video_interpolation_enabled = gr.Checkbox(label = "Enabled", value = False, elem_id = self.elem_id("video_interpolation_enabled"))
                # HACK: Extra space is intentional here, as Web UI "smartly" saves UI parameters using labels as keys, making conflicts inevitable
                ue.video_interpolation_fps = gr.Slider(label = "Frames per second ", minimum = 1, maximum = 60, step = 1, value = 60, elem_id = self.elem_id("video_interpolation_fps"))
                ue.video_interpolation_mb_subframes = gr.Slider(label = "Motion blur subframes", minimum = 0, maximum = 15, step = 1, value = 0, elem_id = self.elem_id("video_interpolation_mb_subframes"))

            with gr.Accordion("Temporal blurring"):
                ue.video_temporal_blurring_enabled = gr.Checkbox(label = "Enabled", value = False, elem_id = self.elem_id("video_temporal_blurring_enabled"))
                # HACK: Extra space is intentional here, as Web UI "smartly" saves UI parameters using labels as keys, making conflicts inevitable
                ue.video_temporal_blurring_radius = gr.Slider(label = "Radius ", minimum = 1, maximum = 10, step = 1, value = 1, elem_id = self.elem_id("video_temporal_blurring_radius"))
                ue.video_temporal_blurring_easing = gr.Slider(label = "Easing", minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0, elem_id = self.elem_id("video_temporal_blurring_easing"))

            with gr.Accordion("Scaling"):
                ue.video_scaling_enabled = gr.Checkbox(label = "Enabled", value = False, elem_id = self.elem_id("video_scaling_enabled"))

                with gr.Row():
                    ue.video_scaling_width = gr.Slider(label = "Width", minimum = 16, maximum = 2560, step = 16, value = 512, elem_id = self.elem_id("video_scaling_width"))
                    ue.video_scaling_height = gr.Slider(label = "Height", minimum = 16, maximum = 2560, step = 16, value = 512, elem_id = self.elem_id("video_scaling_height"))

            with gr.Accordion("Frame number overlay"):
                ue.video_frame_num_overlay_enabled = gr.Checkbox(label = "Enabled", value = False, elem_id = self.elem_id("video_frame_num_overlay_enabled"))
                ue.video_frame_num_overlay_font_size = gr.Number(label = "Font size", precision = 0, minimum = 1, maximum = 144, step = 1, value = 16, elem_id = self.elem_id("video_frame_num_overlay_font_size"))

                with gr.Row():
                    ue.video_frame_num_overlay_text_color = gr.ColorPicker(label = "Text color", value = "#ffffff", elem_id = self.elem_id("video_frame_num_overlay_text_color"))
                    ue.video_frame_num_overlay_text_alpha = gr.Slider(label = "Text alpha", minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0, elem_id = self.elem_id("video_frame_num_overlay_text_alpha"))

                with gr.Row():
                    ue.video_frame_num_overlay_shadow_color = gr.ColorPicker(label = "Shadow color", value = "#000000", elem_id = self.elem_id("video_frame_num_overlay_shadow_color"))
                    ue.video_frame_num_overlay_shadow_alpha = gr.Slider(label = "Shadow alpha", minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0, elem_id = self.elem_id("video_frame_num_overlay_shadow_alpha"))

            with gr.Row():
                ue.render_draft_on_finish = gr.Checkbox(label = "Render draft when finished", value = False, elem_id = self.elem_id("render_draft_on_finish"))
                ue.render_final_on_finish = gr.Checkbox(label = "Render final when finished", value = False, elem_id = self.elem_id("render_final_on_finish"))

            with gr.Row():
                ue.render_draft = gr.Button(value = "Render draft", elem_id = self.elem_id("render_draft"))
                ue.render_final = gr.Button(value = "Render final", elem_id = self.elem_id("render_final"))

            ue.video_preview = gr.Video(label = "Preview", format = "mp4", interactive = False, elem_id = self.elem_id("video_preview"))

        with gr.Tab("Metrics"):
            ue.metrics_enabled = gr.Checkbox(label = "Enabled", value = False, elem_id = self.elem_id("metrics_enabled"))
            ue.metrics_save_plots_every_nth_frame = gr.Number(label = "Save plots every N-th frame", precision = 0, minimum = 1, step = 1, value = 10, elem_id = self.elem_id("metrics_save_plots_every_nth_frame"))
            ue.render_plots = gr.Button(value = "Render plots", elem_id = self.elem_id("render_plots"))
            ue.metrics_plots = gr.Gallery(label = "Plots", columns = 4, object_fit = "contain", preview = True, elem_id = self.elem_id("metrics_plots"))

        with gr.Tab("Help"):
            for file_name, title in [
                ("tab_general.md", "General tab"),
                ("tab_frame_preprocessing.md", "Frame Preprocessing tab"),
                ("tab_video_rendering.md", "Video Rendering tab"),
                ("tab_metrics.md", "Metrics tab"),
                ("additional_notes.md", "Additional notes"),
            ]:
                with open(EXTENSION_DIR / f"docs/temporal/{file_name}", "r", encoding = "utf-8") as file:
                    text = file.read()

                with gr.Accordion(title, open = False):
                    gr.Markdown(text)

        def make_render_callback(is_final):
            def callback(*args):
                yield gr.Button.update(interactive = False), gr.Button.update(interactive = False), None

                self._start_video_render(is_final, *args)

                while self._video_render_tq.busy:
                    sleep(1)

                uv = self._get_ui_values(*args)

                yield gr.Button.update(interactive = True), gr.Button.update(interactive = True), f"{uv.output_dir}/{uv.project_subdir}-{'final' if is_final else 'draft'}.mp4"

            return callback

        def render_plots_callback(*args):
            uv = self._get_ui_values(*args)
            project_dir = Path(uv.output_dir) / uv.project_subdir
            metrics = Metrics()
            metrics.load(project_dir)
            return gr.Gallery.update(value = list(metrics.plot(project_dir)))

        ue.render_draft.click(make_render_callback(False), inputs = list(ue_dict.values()), outputs = [ue.render_draft, ue.render_final, ue.video_preview])
        ue.render_final.click(make_render_callback(True), inputs = list(ue_dict.values()), outputs = [ue.render_draft, ue.render_final, ue.video_preview])
        ue.render_plots.click(render_plots_callback, inputs = list(ue_dict.values()), outputs = [ue.metrics_plots])

        self._ui_element_names = list(ue_dict.keys())

        return list(ue_dict.values())

    def run(self, p, *args):
        uv = self._get_ui_values(*args)
        metrics = Metrics()

        project_dir = safe_get_directory(Path(uv.output_dir) / uv.project_subdir)
        session_dir = safe_get_directory(project_dir / "session")

        if uv.start_from_scratch:
            for path in project_dir.glob("*.png"):
                path.unlink()

            metrics.clear(project_dir)

        last_index = get_last_frame_index(project_dir)

        p.prompt = prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        p.negative_prompt = prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
        p.styles.clear()

        if uv.load_session:
            load_session(p, uv, project_dir, session_dir, last_index)

        if uv.metrics_enabled:
            metrics.load(project_dir)

        p.n_iter = 1
        p.batch_size = 1
        p.do_not_save_samples = True
        p.do_not_save_grid = True
        processing.fix_seed(p)

        if not p.init_images or not isinstance(p.init_images[0], Image.Image):
            if processed := generate_image(
                "Initial image",
                p,
                init_images = [generate_noise_image((p.width, p.height), p.seed)],
                denoising_strength = 1.0,
            ):
                p.init_images = [processed.images[0]]
                p.seed += 1
            else:
                return processed

        if uv.metrics_enabled and last_index == 0:
            metrics.measure(p.init_images[0])

        if opts.img2img_color_correction:
            p.color_corrections = [processing.setup_color_correction(p.init_images[0])]

        if uv.save_session:
            save_session(p, uv, project_dir, session_dir, last_index)

        if uv.noise_relative:
            uv.noise_amount *= p.denoising_strength

        if uv.modulation_relative:
            uv.modulation_amount *= p.denoising_strength

        if uv.tinting_relative:
            uv.tinting_amount *= p.denoising_strength

        if uv.sharpening_relative:
            uv.sharpening_amount *= p.denoising_strength

        state.job_count = uv.frame_count

        last_image = p.init_images[0]
        last_seed = p.seed

        for i, frame_index in zip(range(uv.frame_count), count(last_index + 1)):
            if not (processed := generate_image(
                f"Frame {i + 1} / {uv.frame_count}",
                p,
                init_images = [preprocess_image(last_image, uv, last_seed)],
                seed = last_seed,
            )):
                return processed

            last_image = processed.images[0]
            last_seed += 1

            if frame_index % uv.save_every_nth_frame == 0:
                if uv.archive_mode:
                    self._image_save_tq.enqueue(
                        Image.Image.save,
                        last_image,
                        project_dir / f"{frame_index:05d}.png",
                        optimize = True,
                        compress_level = 9,
                    )
                else:
                    images.save_image(
                        last_image,
                        project_dir,
                        "",
                        processed.seed,
                        p.prompt,
                        opts.samples_format,
                        info = processed.info,
                        p = p,
                        forced_filename = f"{frame_index:05d}",
                    )

            if uv.metrics_enabled:
                metrics.measure(last_image)
                metrics.save(project_dir)

                if frame_index % uv.metrics_save_plots_every_nth_frame == 0:
                    metrics.plot(project_dir, save_images = True)

        if uv.render_draft_on_finish:
            self._start_video_render(False, *args)

        if uv.render_final_on_finish:
            self._start_video_render(True, *args)

        return processed

    def _get_ui_values(self, *args):
        return SimpleNamespace(**{name: arg for name, arg in zip(self._ui_element_names, args)})

    def _start_video_render(self, is_final, *args):
        self._video_render_tq.enqueue(render_video, self._get_ui_values(*args), is_final)
