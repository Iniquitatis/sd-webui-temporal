import json
from copy import copy
from importlib import import_module
from itertools import count
from os import system
from pathlib import Path
from threading import Lock, Thread
from types import SimpleNamespace

import gradio as gr
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
from skimage.exposure import match_histograms

from modules import images, processing, scripts
from modules.shared import opts, prompt_styles, state

external_code = import_module("extensions.sd-webui-controlnet.scripts.external_code", "external_code")

#===============================================================================

def safe_get_directory(path):
    path.mkdir(parents = True, exist_ok = True)
    return path

#===============================================================================

class ThreadQueue:
    def __init__(self):
        self._threads = []
        self._start_lock = Lock()
        self._list_lock = Lock()

    def enqueue(self, target, *args, **kwargs):
        def callback():
            with self._start_lock:
                target(*args, **kwargs)

            with self._list_lock:
                self._threads = list(filter(Thread.is_alive, self._threads))

        with self._list_lock:
            thread = Thread(target = callback)
            thread.start()
            self._threads.append(thread)

#===============================================================================

BLEND_MODES = dict(
    normal     = dict(name = "Normal",     func = lambda im, mod, amt: Image.blend(im, mod, amt)),
    add        = dict(name = "Add",        func = lambda im, mod, amt: Image.blend(im, ImageChops.add(im, mod), amt)),
    subtract   = dict(name = "Subtract",   func = lambda im, mod, amt: Image.blend(im, ImageChops.subtract(im, mod), amt)),
    multiply   = dict(name = "Multiply",   func = lambda im, mod, amt: Image.blend(im, ImageChops.multiply(im, mod), amt)),
    lighten    = dict(name = "Lighten",    func = lambda im, mod, amt: Image.blend(im, ImageChops.lighter(im, mod), amt)),
    darken     = dict(name = "Darken",     func = lambda im, mod, amt: Image.blend(im, ImageChops.darker(im, mod), amt)),
    hard_light = dict(name = "Hard light", func = lambda im, mod, amt: Image.blend(im, ImageChops.hard_light(im, mod), amt)),
    soft_light = dict(name = "Soft light", func = lambda im, mod, amt: Image.blend(im, ImageChops.soft_light(im, mod), amt)),
    overlay    = dict(name = "Overlay",    func = lambda im, mod, amt: Image.blend(im, ImageChops.overlay(im, mod), amt)),
    screen     = dict(name = "Screen",     func = lambda im, mod, amt: Image.blend(im, ImageChops.screen(im, mod), amt)),
    # TODO: Rename to monochrome/color/whatever
    tint       = dict(name = "Tint",       func = lambda im, mod, amt: Image.blend(im, ImageChops.multiply(ImageOps.grayscale(im).convert(im.mode), mod), amt)),
)

def blend_images(im, modulator, mode, amount):
    if modulator is None or amount <= 0.0:
        return im

    return BLEND_MODES[mode]["func"](im, modulator, amount)

def generate_noise_image(size, seed):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, size = (size[1], size[0], 3), dtype = "uint8"))

def load_image(path):
    im = Image.open(path)
    im.load()
    return im

def preprocess_image(im, uv, seed):
    if uv.color_correction_image is not None:
        im = Image.fromarray(match_histograms(np.array(im), np.array(uv.color_correction_image.convert(im.mode)), channel_axis = 2).astype("uint8"))

    if uv.normalize_contrast:
        im = ImageOps.autocontrast(im.convert("RGB"), preserve_tone = True)

    if uv.brightness != 1.0:
        im = ImageEnhance.Brightness(im).enhance(uv.brightness)

    if uv.contrast != 1.0:
        im = ImageEnhance.Contrast(im).enhance(uv.contrast)

    if uv.saturation != 1.0:
        im = ImageEnhance.Color(im).enhance(uv.saturation)

    if uv.noise_amount > 0.0:
        im = blend_images(im, generate_noise_image(im.size, seed), uv.noise_mode, uv.noise_amount)

    if uv.modulator_image is not None and uv.modulator_amount > 0.0:
        im = blend_images(im, (
            uv.modulator_image
            .convert(im.mode)
            .resize(im.size, Image.Resampling.LANCZOS)
            .filter(ImageFilter.GaussianBlur(uv.modulator_blurring))
        ), uv.modulator_mode, uv.modulator_amount)

    if uv.contour_image is not None and uv.contour_amount > 0.0:
        im = blend_images(im, ImageOps.grayscale((
            uv.contour_image
            .convert(im.mode)
            .resize(im.size, Image.Resampling.LANCZOS)
            .filter(ImageFilter.CONTOUR)
            .filter(ImageFilter.GaussianBlur(uv.contour_blurring))
        )).convert(im.mode), "multiply", uv.contour_amount)

    if uv.tinting_color != "#ffffff" and uv.tinting_amount > 0.0:
        im = blend_images(im, Image.new(im.mode, im.size, uv.tinting_color), uv.tinting_mode, uv.tinting_amount)

    if uv.sharpening_amount > 0.0:
        im = im.filter(ImageFilter.UnsharpMask(uv.sharpening_radius, int(uv.sharpening_amount * 100.0), 0))

    if uv.zooming > 0:
        pixel_size = 1.0 / (im.width if im.width > im.height else im.height)
        zoom_factor = 1.0 - pixel_size * uv.zooming
        im = im.transform(im.size, Image.AFFINE, (
            zoom_factor, 0.0, -((im.width  * zoom_factor) - im.width)  / 2.0,
            0.0, zoom_factor, -((im.height * zoom_factor) - im.height) / 2.0,
        ), Image.Resampling.BILINEAR)

    if uv.shifting_x > 0 or uv.shifting_y > 0:
        im = ImageChops.offset(im, uv.shifting_x, uv.shifting_y)

    if uv.symmetry:
        im.paste(ImageOps.mirror(im.crop((0, 0, im.width // 2, im.height))), (im.width // 2, 0))

    if uv.blurring > 0.0:
        im = im.filter(ImageFilter.GaussianBlur(uv.blurring))

    if uv.spreading > 0:
        # TODO: Make deterministic by making use of the provided seed
        im = im.effect_spread(uv.spreading)

    return im

#===============================================================================

INTERPOLATED_FPS = 60

def render_video(uv, is_final):
    suffix = "final" if is_final else "draft"

    output_dir = Path(uv.output_dir)
    frame_dir = output_dir / uv.project_subdir
    frame_list_path = output_dir / f"{uv.project_subdir}-{suffix}.txt"
    video_path = output_dir / f"{uv.project_subdir}-{suffix}.mkv"

    with open(frame_list_path, "w", encoding = "utf-8") as frame_list:
        frame_paths = sorted(frame_dir.glob("*.png"), key = lambda x: x.name)

        if uv.video_looping:
            frame_paths += reversed(frame_paths[:-1])

        for frame_path in frame_paths:
            frame_list.write(f"file '{Path(uv.project_subdir) / frame_path.name}'\nduration 1\n")

    filters = []

    if is_final:
        if uv.video_deflickering:
            filters.append(f"deflicker='size={uv.video_fps}:mode=am'")

        if uv.video_interpolation:
            filters.append(f"minterpolate='fps={INTERPOLATED_FPS * (uv.video_mb_subframes + 1)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=none'")

            if uv.video_mb_subframes > 0:
                filters.append(f"tmix='frames={uv.video_mb_subframes + 1}'")
                filters.append(f"fps={INTERPOLATED_FPS}")

        filters.append(f"scale='{uv.video_width}x{uv.video_height}:flags=lanczos'")

    if uv.video_frame_num_overlay:
        filters.append(f"drawtext='text=\"%{{eif\\:n*{uv.video_fps / (INTERPOLATED_FPS * (uv.video_mb_subframes + 1)) if is_final and uv.video_interpolation else 1.0:.18f}+1\\:d\\:5}}\":x=5:y=5:fontsize=16:fontcolor=#ffffffc0:shadowx=1:shadowy=1:shadowcolor=#000000c0'")

    system(" ".join([
        f"ffmpeg",
        f"-y",
        f"-r {uv.video_fps}",
        f"-f concat",
        f"-safe 0",
        f"-i \"{frame_list_path}\"",
        f"-framerate {uv.video_fps}",
        f"-vf {','.join(filters)}" if len(filters) > 0 else "",
        f"-c:v libx264",
        f"-crf 14",
        f"-preset {'slow' if is_final else 'veryfast'}",
        f"-tune film",
        f"-pix_fmt yuv420p",
        f"\"{video_path}\"",
    ]))

    frame_list_path.unlink()

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

def load_session(p, uv, project_dir, session_dir, params_path, last_index):
    if not params_path.is_file():
        return

    with open(params_path, "r", encoding = "utf-8") as params_file:
        data = json.load(params_file)

    load_object(p, data.get("generation_params", {}), session_dir)

    for unit_data, cn_unit in zip(data.get("controlnet_params", []), external_code.get_all_units_in_processing(p)):
        load_object(cn_unit, unit_data, session_dir)

    load_object(uv, data.get("extension_params", {}), session_dir)

    if (im_path := (project_dir / f"{last_index:05d}.png")).is_file():
        p.init_images = [load_image(im_path)]

    # FIXME: Unreliable; works properly only on first generation, but not on the subsequent ones
    if p.seed != -1:
        p.seed = p.seed + last_index

def save_session(p, uv, project_dir, session_dir, params_path, last_index):
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
        ),
        extension_params = save_object(uv, session_dir, [
            "save_every_nth_frame",
            "color_correction_image",
            "normalize_contrast",
            "brightness",
            "contrast",
            "saturation",
            "noise_mode",
            "noise_amount",
            "noise_relative",
            "modulator_image",
            "modulator_blurring",
            "modulator_mode",
            "modulator_amount",
            "modulator_relative",
            "contour_image",
            "contour_blurring",
            "contour_amount",
            "contour_relative",
            "tinting_color",
            "tinting_mode",
            "tinting_amount",
            "tinting_relative",
            "sharpening_radius",
            "sharpening_amount",
            "sharpening_relative",
            "zooming",
            "shifting_x",
            "shifting_y",
            "symmetry",
            "blurring",
            "spreading",
        ]),
    )

    with open(params_path, "w", encoding = "utf-8") as params_file:
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
            ue.color_correction_image = gr.Pil(label = "Color correction image", elem_id = self.elem_id("color_correction_image"))
            ue.normalize_contrast = gr.Checkbox(label = "Normalize contrast", value = False, elem_id = self.elem_id("normalize_contrast"))
            ue.brightness = gr.Slider(label = "Brightness", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, elem_id = self.elem_id("brightness"))
            ue.contrast = gr.Slider(label = "Contrast", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, elem_id = self.elem_id("contrast"))
            ue.saturation = gr.Slider(label = "Saturation", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, elem_id = self.elem_id("saturation"))
            # FIXME: Pairs (name, value) don't work for some reason
            ue.noise_mode = gr.Dropdown(label = "Noise mode", type = "value", choices = list(BLEND_MODES.keys()), value = next(iter(BLEND_MODES)), elem_id = self.elem_id("noise_mode"))
            ue.noise_amount = gr.Slider(label = "Noise amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, elem_id = self.elem_id("noise_amount"))
            ue.noise_relative = gr.Checkbox(label = "Noise relative", value = False, elem_id = self.elem_id("noise_relative"))
            ue.modulator_image = gr.Pil(label = "Modulator image", elem_id = self.elem_id("modulator_image"))
            ue.modulator_blurring = gr.Slider(label = "Modulator blurring", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0, elem_id = self.elem_id("modulator_blurring"))
            # FIXME: Pairs (name, value) don't work for some reason
            ue.modulator_mode = gr.Dropdown(label = "Modulator mode", type = "value", choices = list(BLEND_MODES.keys()), value = next(iter(BLEND_MODES)), elem_id = self.elem_id("modulator_mode"))
            ue.modulator_amount = gr.Slider(label = "Modulator amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, elem_id = self.elem_id("modulator_amount"))
            ue.modulator_relative = gr.Checkbox(label = "Modulator relative", value = False, elem_id = self.elem_id("modulator_relative"))
            ue.contour_image = gr.Pil(label = "Contour image", elem_id = self.elem_id("contour_image"))
            ue.contour_blurring = gr.Slider(label = "Contour blurring", minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0, elem_id = self.elem_id("contour_blurring"))
            ue.contour_amount = gr.Slider(label = "Contour amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, elem_id = self.elem_id("contour_amount"))
            ue.contour_relative = gr.Checkbox(label = "Contour relative", value = False, elem_id = self.elem_id("contour_relative"))
            ue.tinting_color = gr.ColorPicker(label = "Tinting color", value = "#ffffff", elem_id = self.elem_id("tinting_color"))
            # FIXME: Pairs (name, value) don't work for some reason
            ue.tinting_mode = gr.Dropdown(label = "Tinting mode", type = "value", choices = list(BLEND_MODES.keys()), value = next(iter(BLEND_MODES)), elem_id = self.elem_id("tinting_mode"))
            ue.tinting_amount = gr.Slider(label = "Tinting amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, elem_id = self.elem_id("tinting_amount"))
            ue.tinting_relative = gr.Checkbox(label = "Tinting relative", value = False, elem_id = self.elem_id("tinting_relative"))
            ue.sharpening_radius = gr.Slider(label = "Sharpening radius", minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0, elem_id = self.elem_id("sharpening_radius"))
            ue.sharpening_amount = gr.Slider(label = "Sharpening amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, elem_id = self.elem_id("sharpening_amount"))
            ue.sharpening_relative = gr.Checkbox(label = "Sharpening relative", value = False, elem_id = self.elem_id("sharpening_relative"))
            ue.zooming = gr.Slider(label = "Zooming", minimum = 0, maximum = 10, step = 1, value = 0, elem_id = self.elem_id("zooming"))

            with gr.Row():
                ue.shifting_x = gr.Number(label = "Shifting X", precision = 0, step = 1, value = 0, elem_id = self.elem_id("shifting_x"))
                ue.shifting_y = gr.Number(label = "Shifting Y", precision = 0, step = 1, value = 0, elem_id = self.elem_id("shifting_y"))

            ue.symmetry = gr.Checkbox(label = "Symmetry", value = False, elem_id = self.elem_id("symmetry"))
            ue.blurring = gr.Slider(label = "Blurring", minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0, elem_id = self.elem_id("blurring"))
            ue.spreading = gr.Slider(label = "Spreading", minimum = 0, maximum = 10, step = 1, value = 0, elem_id = self.elem_id("spreading"))

        with gr.Tab("Video Rendering"):
            with gr.Row():
                ue.video_width = gr.Slider(label = "Width", minimum = 16, maximum = 2560, step = 16, value = 1024, elem_id = self.elem_id("video_width"))
                ue.video_height = gr.Slider(label = "Height", minimum = 16, maximum = 2560, step = 16, value = 576, elem_id = self.elem_id("video_height"))

            ue.video_fps = gr.Slider(label = "Frames per second", minimum = 1, maximum = 60, step = 1, value = 30, elem_id = self.elem_id("video_fps"))
            ue.video_interpolation = gr.Checkbox(label = "Interpolation", value = False, elem_id = self.elem_id("video_interpolation"))
            ue.video_mb_subframes = gr.Slider(label = "Motion blur subframes", minimum = 0, maximum = 15, step = 1, value = 0, elem_id = self.elem_id("video_mb_subframes"))
            ue.video_deflickering = gr.Checkbox(label = "Deflickering", value = True, elem_id = self.elem_id("video_deflickering"))
            ue.video_looping = gr.Checkbox(label = "Looping", value = False, elem_id = self.elem_id("video_looping"))
            ue.video_frame_num_overlay = gr.Checkbox(label = "Frame number overlay", value = False, elem_id = self.elem_id("video_frame_num_overlay"))

            with gr.Row():
                ue.render_draft_on_finish = gr.Checkbox(label = "Render draft when finished", value = False, elem_id = self.elem_id("render_draft_on_finish"))
                ue.render_final_on_finish = gr.Checkbox(label = "Render final when finished", value = False, elem_id = self.elem_id("render_final_on_finish"))

            with gr.Row():
                ue.render_draft = gr.Button(value = "Render draft", elem_id = self.elem_id("render_draft"))
                ue.render_final = gr.Button(value = "Render final", elem_id = self.elem_id("render_final"))

        ue.render_draft.click(lambda *args: self._start_video_render(False, *args), inputs = list(ue_dict.values()), outputs = [])
        ue.render_final.click(lambda *args: self._start_video_render(True, *args), inputs = list(ue_dict.values()), outputs = [])

        self._ui_element_names = list(ue_dict.keys())

        return list(ue_dict.values())

    def run(self, p, *args):
        uv = self._get_ui_values(*args)

        project_dir = safe_get_directory(Path(uv.output_dir) / uv.project_subdir)
        session_dir = safe_get_directory(project_dir / "session")
        params_path = session_dir / "parameters.json"

        if uv.start_from_scratch:
            for path in project_dir.rglob("*.png"):
                path.unlink()

        last_index = get_last_frame_index(project_dir)

        p.prompt = prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        p.negative_prompt = prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
        p.styles.clear()

        if uv.load_session:
            load_session(p, uv, project_dir, session_dir, params_path, last_index)

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

        if opts.img2img_color_correction:
            p.color_corrections = [processing.setup_color_correction(p.init_images[0])]

        if uv.save_session:
            save_session(p, uv, project_dir, session_dir, params_path, last_index)

        if uv.noise_relative:
            uv.noise_amount *= p.denoising_strength

        if uv.modulator_relative:
            uv.modulator_amount *= p.denoising_strength

        if uv.contour_relative:
            uv.contour_amount *= p.denoising_strength

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

            if frame_index % uv.save_every_nth_frame == 0:
                if uv.archive_mode:
                    self._image_save_tq.enqueue(
                        Image.Image.save,
                        processed.images[0],
                        project_dir / f"{frame_index:05d}.png",
                        optimize = True,
                        compress_level = 9,
                    )
                else:
                    images.save_image(
                        processed.images[0],
                        project_dir,
                        "",
                        processed.seed,
                        p.prompt,
                        opts.samples_format,
                        info = processed.info,
                        p = p,
                        forced_filename = f"{frame_index:05d}",
                    )

            last_image = processed.images[0]
            last_seed += 1

        if uv.render_draft_on_finish:
            self._start_video_render(False, *args)

        if uv.render_final_on_finish:
            self._start_video_render(True, *args)

        return processed

    def _get_ui_values(self, *args):
        return SimpleNamespace(**{name: arg for name, arg in zip(self._ui_element_names, args)})

    def _start_video_render(self, is_final, *args):
        self._video_render_tq.enqueue(render_video, self._get_ui_values(*args), is_final)
