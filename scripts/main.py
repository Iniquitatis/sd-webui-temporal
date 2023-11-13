from copy import copy
from itertools import count
from pathlib import Path
from subprocess import run
from time import sleep
from types import SimpleNamespace

import gradio as gr
import numpy as np
import scipy
import skimage
from PIL import Image, ImageColor

from modules import images, processing, scripts
from modules.shared import opts, prompt_styles, state

from temporal.blend_modes import BLEND_MODES, blend_images
from temporal.fs import safe_get_directory
from temporal.image_utils import generate_noise_image, match_image
from temporal.interop import EXTENSION_DIR
from temporal.math import remap_range
from temporal.metrics import Metrics
from temporal.session import get_last_frame_index, load_session, save_session
from temporal.thread_queue import ThreadQueue

#===============================================================================

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

def generate_image(job_title, p, **p_overrides):
    state.job = job_title

    p_instance = copy(p)

    for key, value in p_overrides.items():
        if hasattr(p_instance, key):
            setattr(p_instance, key, value)
        else:
            print(f"WARNING: Key {key} doesn't exist in {p_instance.__class__.__name__}")

    try:
        processed = processing.process_images(p_instance)
    except Exception:
        return None

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

        labels = set()

        def unique_label(string):
            if string in labels:
                string = unique_label(string + " ")

            labels.add(string)

            return string

        def elem(key, gr_type, *args, **kwargs):
            if "label" in kwargs:
                kwargs["label"] = unique_label(kwargs["label"])

            elem = gr_type(*args, elem_id = self.elem_id(key), **kwargs)
            setattr(ue, key, elem)

            return elem

        with gr.Tab("General"):
            elem("output_dir", gr.Textbox, label = "Output directory", value = "outputs/temporal")
            elem("project_subdir", gr.Textbox, label = "Project subdirectory", value = "untitled")
            elem("frame_count", gr.Number, label = "Frame count", precision = 0, minimum = 1, step = 1, value = 100)
            elem("save_every_nth_frame", gr.Number, label = "Save every N-th frame", precision = 0, minimum = 1, step = 1, value = 1)
            elem("archive_mode", gr.Checkbox, label = "Archive mode", value = False)
            elem("start_from_scratch", gr.Checkbox, label = "Start from scratch", value = False)
            elem("load_session", gr.Checkbox, label = "Load session", value = True)
            elem("save_session", gr.Checkbox, label = "Save session", value = True)

        with gr.Tab("Frame Preprocessing"):
            with gr.Accordion("Noise compression"):
                elem("noise_compression_enabled", gr.Checkbox, label = "Enabled", value = False)
                elem("noise_compression_constant", gr.Slider, label = "Constant", minimum = 0.0, maximum = 1.0, step = 1e-5, value = 0.0)
                elem("noise_compression_adaptive", gr.Slider, label = "Adaptive", minimum = 0.0, maximum = 2.0, step = 0.01, value = 0.0)

            with gr.Accordion("Color correction"):
                elem("color_correction_enabled", gr.Checkbox, label = "Enabled", value = False)
                elem("color_correction_image", gr.Pil, label = "Reference image")
                elem("normalize_contrast", gr.Checkbox, label = "Normalize contrast", value = False)

            with gr.Accordion("Color balancing"):
                elem("color_balancing_enabled", gr.Checkbox, label = "Enabled", value = False)
                elem("brightness", gr.Slider, label = "Brightness", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0)
                elem("contrast", gr.Slider, label = "Contrast", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0)
                elem("saturation", gr.Slider, label = "Saturation", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0)

            with gr.Accordion("Noise"):
                elem("noise_enabled", gr.Checkbox, label = "Enabled", value = False)

                with gr.Row():
                    elem("noise_amount", gr.Slider, label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0)
                    elem("noise_relative", gr.Checkbox, label = "Relative", value = False)

                # FIXME: Pairs (name, value) don't work in older versions of Gradio
                elem("noise_mode", gr.Dropdown, label = "Mode", type = "value", choices = list(BLEND_MODES.keys()), value = next(iter(BLEND_MODES)))

            with gr.Accordion("Modulation"):
                elem("modulation_enabled", gr.Checkbox, label = "Enabled", value = False)

                with gr.Row():
                    elem("modulation_amount", gr.Slider, label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0)
                    elem("modulation_relative", gr.Checkbox, label = "Relative", value = False)

                # FIXME: Pairs (name, value) don't work in older versions of Gradio
                elem("modulation_mode", gr.Dropdown, label = "Mode", type = "value", choices = list(BLEND_MODES.keys()), value = next(iter(BLEND_MODES)))
                elem("modulation_image", gr.Pil, label = "Image")
                elem("modulation_blurring", gr.Slider, label = "Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0)

            with gr.Accordion("Tinting"):
                elem("tinting_enabled", gr.Checkbox, label = "Enabled", value = False)

                with gr.Row():
                    elem("tinting_amount", gr.Slider, label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0)
                    elem("tinting_relative", gr.Checkbox, label = "Relative", value = False)

                # FIXME: Pairs (name, value) don't work in older versions of Gradio
                elem("tinting_mode", gr.Dropdown, label = "Mode", type = "value", choices = list(BLEND_MODES.keys()), value = next(iter(BLEND_MODES)))
                elem("tinting_color", gr.ColorPicker, label = "Color", value = "#ffffff")

            with gr.Accordion("Sharpening"):
                elem("sharpening_enabled", gr.Checkbox, label = "Enabled", value = False)

                with gr.Row():
                    elem("sharpening_amount", gr.Slider, label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0)
                    elem("sharpening_relative", gr.Checkbox, label = "Relative", value = False)

                elem("sharpening_radius", gr.Slider, label = "Radius", minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0)

            with gr.Accordion("Transformation"):
                elem("transformation_enabled", gr.Checkbox, label = "Enabled", value = False)

                with gr.Row():
                    elem("translation_x", gr.Number, label = "Translation X", step = 0.001, value = 0.0)
                    elem("translation_y", gr.Number, label = "Translation Y", step = 0.001, value = 0.0)

                elem("rotation", gr.Slider, label = "Rotation", minimum = -90.0, maximum = 90.0, step = 0.1, value = 0.0)
                elem("scaling", gr.Slider, label = "Scaling", minimum = 0.0, maximum = 2.0, step = 0.001, value = 1.0)

            elem("symmetrize", gr.Checkbox, label = "Symmetrize", value = False)

            with gr.Accordion("Blurring"):
                elem("blurring_enabled", gr.Checkbox, label = "Enabled", value = False)
                elem("blurring_radius", gr.Slider, label = "Radius", minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0)

            with gr.Accordion("Custom code"):
                elem("custom_code_enabled", gr.Checkbox, label = "Enabled", value = False)
                gr.Markdown("**WARNING:** Don't put an untrusted code here!")
                elem("custom_code", gr.Code, label = "Code", language = "python", value = "")

        with gr.Tab("Video Rendering"):
            elem("video_fps", gr.Slider, label = "Frames per second", minimum = 1, maximum = 60, step = 1, value = 30)
            elem("video_looping", gr.Checkbox, label = "Looping", value = False)

            with gr.Accordion("Deflickering"):
                elem("video_deflickering_enabled", gr.Checkbox, label = "Enabled", value = False)
                elem("video_deflickering_frames", gr.Slider, label = "Frames", minimum = 2, maximum = 120, step = 1, value = 60)

            with gr.Accordion("Interpolation"):
                elem("video_interpolation_enabled", gr.Checkbox, label = "Enabled", value = False)
                elem("video_interpolation_fps", gr.Slider, label = "Frames per second", minimum = 1, maximum = 60, step = 1, value = 60)
                elem("video_interpolation_mb_subframes", gr.Slider, label = "Motion blur subframes", minimum = 0, maximum = 15, step = 1, value = 0)

            with gr.Accordion("Temporal blurring"):
                elem("video_temporal_blurring_enabled", gr.Checkbox, label = "Enabled", value = False)
                elem("video_temporal_blurring_radius", gr.Slider, label = "Radius", minimum = 1, maximum = 10, step = 1, value = 1)
                elem("video_temporal_blurring_easing", gr.Slider, label = "Easing", minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0)

            with gr.Accordion("Scaling"):
                elem("video_scaling_enabled", gr.Checkbox, label = "Enabled", value = False)

                with gr.Row():
                    elem("video_scaling_width", gr.Slider, label = "Width", minimum = 16, maximum = 2560, step = 16, value = 512)
                    elem("video_scaling_height", gr.Slider, label = "Height", minimum = 16, maximum = 2560, step = 16, value = 512)

            with gr.Accordion("Frame number overlay"):
                elem("video_frame_num_overlay_enabled", gr.Checkbox, label = "Enabled", value = False)
                elem("video_frame_num_overlay_font_size", gr.Number, label = "Font size", precision = 0, minimum = 1, maximum = 144, step = 1, value = 16)

                with gr.Row():
                    elem("video_frame_num_overlay_text_color", gr.ColorPicker, label = "Text color", value = "#ffffff")
                    elem("video_frame_num_overlay_text_alpha", gr.Slider, label = "Text alpha", minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0)

                with gr.Row():
                    elem("video_frame_num_overlay_shadow_color", gr.ColorPicker, label = "Shadow color", value = "#000000")
                    elem("video_frame_num_overlay_shadow_alpha", gr.Slider, label = "Shadow alpha", minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0)

            with gr.Row():
                elem("render_draft_on_finish", gr.Checkbox, label = "Render draft when finished", value = False)
                elem("render_final_on_finish", gr.Checkbox, label = "Render final when finished", value = False)

            with gr.Row():
                elem("render_draft", gr.Button, value = "Render draft")
                elem("render_final", gr.Button, value = "Render final")

            elem("video_preview", gr.Video, label = "Preview", format = "mp4", interactive = False)

        with gr.Tab("Metrics"):
            elem("metrics_enabled", gr.Checkbox, label = "Enabled", value = False)
            elem("metrics_save_plots_every_nth_frame", gr.Number, label = "Save plots every N-th frame", precision = 0, minimum = 1, step = 1, value = 10)
            elem("render_plots", gr.Button, value = "Render plots")
            elem("metrics_plots", gr.Gallery, label = "Plots", columns = 4, object_fit = "contain", preview = True)

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

        opts_backup = opts.data.copy()

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
                return processing.Processed(p, p.init_images)

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
                processed = processing.Processed(p, [last_image])
                break

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

        opts.data.update(opts_backup)

        return processed

    def _get_ui_values(self, *args):
        return SimpleNamespace(**{name: arg for name, arg in zip(self._ui_element_names, args)})

    def _start_video_render(self, is_final, *args):
        self._video_render_tq.enqueue(render_video, self._get_ui_values(*args), is_final)
