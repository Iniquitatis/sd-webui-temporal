from copy import copy
from itertools import count
from os import system
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import gradio as gr
from PIL import Image, ImageChops, ImageFilter, ImageOps

import modules.scripts as scripts
from modules import images, processing
from modules.shared import opts, state

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
    tint       = dict(name = "Tint",       func = lambda im, mod, amt: Image.blend(im, ImageChops.multiply(ImageOps.grayscale(im).convert(im.mode), mod), amt)),
)

def blend_images(im, modulator, mode, amount):
    if modulator is None or amount <= 0.0:
        return im

    return BLEND_MODES[mode]["func"](im, modulator, amount)

def get_last_frame_index(frame_dir):
    def get_index(path):
        try:
            if path.exists():
                return int(path.stem)
        except:
            print(f"WARNING: {path} doesn't match the frame name format")
            return 0

    return max((get_index(path) for path in frame_dir.glob("*.png")), default = 0)

def preprocess_image(im, uv):
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
        im = blend_images(im, Image.new(im.mode, im.size, uv.tinting_color), "tint", uv.tinting_amount)

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
        im = im.effect_spread(uv.spreading)

    return im

def render_video(uv, is_final):
    suffix = "final" if is_final else "draft"

    output_dir = Path(uv.output_dir)
    frame_dir = output_dir / uv.project_subdir
    frame_list_path = output_dir / f"{uv.project_subdir}-{suffix}.txt"
    video_path = output_dir / f"{uv.project_subdir}-{suffix}.mkv"

    with open(frame_list_path, "w", encoding = "utf-8") as frame_list:
        frame_paths = sorted(frame_dir.glob("*.png"), key = lambda x: x.name)

        if uv.looping:
            frame_paths += reversed(frame_paths[:-1])

        for frame_path in frame_paths:
            frame_list.write(f"file '{frame_path.name}'\nduration 1\n")

    filters = []

    if is_final:
        if uv.video_deflickering:
            filters.append(f"deflicker='size={uv.video_fps}:mode=am'")

        if uv.video_interpolation:
            filters.append(f"minterpolate='fps={60 * (uv.video_mb_subframes + 1)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=none'")

            if uv.video_mb_subframes > 0:
                filters.append(f"tmix='frames={uv.video_mb_subframes + 1}'")
                filters.append(f"fps=60")

        filters.append(f"scale='{uv.video_width}x{uv.video_height}:flags=lanczos'")

    # FIXME: Drops actual frames, not the interpolated ones (works without interpolation; probably will have to resort to ffprobe and cut in the second pass)
    #if uv.looping:
    #    filters.append(f"trim='end_frame={len(frame_paths) * ((60 if uv.video_interpolation else uv.video_fps) / uv.video_fps) - 1}'")

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
        f"-preset slow",
        f"-tune film",
        f"-pix_fmt yuv420p",
        f"\"{video_path}\"",
    ]))

    frame_list_path.unlink()

class TemporalScript(scripts.Script):
    def title(self):
        return "Temporal"

    def show(self, is_img2img):
        return is_img2img
        # FIXME: Doesn't trigger the `run` method
        #return scripts.AlwaysVisible if is_img2img else False

    def ui(self, is_img2img):
        ue = SimpleNamespace()

        with gr.Accordion("Temporal", open = False):
            ue.enable = gr.Checkbox(label = "Enable", value = False, elem_id = self.elem_id("enable"))

            with gr.Tab("General"):
                ue.output_dir = gr.Textbox(label = "Output directory", value = "outputs/temporal", elem_id = self.elem_id("output_dir"))
                ue.project_subdir = gr.Textbox(label = "Project subdirectory", value = "untitled", elem_id = self.elem_id("project_subdir"))
                ue.frame_count = gr.Number(label = "Frame count", precision = 0, minimum = 1, step = 1, value = 100, elem_id = self.elem_id("frame_count"))
                ue.generate_first_frame = gr.Checkbox(label = "Generate first frame", value = False, elem_id = self.elem_id("generate_first_frame"))
                ue.resume = gr.Checkbox(label = "Resume", value = True, elem_id = self.elem_id("resume"))

            with gr.Tab("Frame Preprocessing"):
                ue.modulator_image = gr.Pil(label = "Modulator image", elem_id = self.elem_id("modulator_image"))
                ue.modulator_blurring = gr.Slider(label = "Modulator blurring", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0, elem_id = self.elem_id("modulator_blurring"))
                ue.modulator_mode = gr.Dropdown(label = "Modulator mode", type = "value", choices = [
                    # FIXME: Pairs (name, value) don't work for some reason
                    value
                    for value, definition in BLEND_MODES.items()
                ], value = next(iter(BLEND_MODES)), elem_id = self.elem_id("modulator_mode"))
                ue.modulator_amount = gr.Slider(label = "Modulator amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, elem_id = self.elem_id("modulator_amount"))
                ue.contour_image = gr.Pil(label = "Contour image", elem_id = self.elem_id("contour_image"))
                ue.contour_blurring = gr.Slider(label = "Contour blurring", minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0, elem_id = self.elem_id("contour_blurring"))
                ue.contour_amount = gr.Slider(label = "Contour amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, elem_id = self.elem_id("contour_amount"))
                ue.tinting_color = gr.ColorPicker(label = "Tinting color", value = "#ffffff", elem_id = self.elem_id("tinting_color"))
                ue.tinting_amount = gr.Slider(label = "Tinting amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, elem_id = self.elem_id("tinting_amount"))
                ue.sharpening_radius = gr.Slider(label = "Sharpening radius", minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0, elem_id = self.elem_id("sharpening_radius"))
                ue.sharpening_amount = gr.Slider(label = "Sharpening amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, elem_id = self.elem_id("sharpening_amount"))
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

                with gr.Row():
                    ue.render_draft_on_finish = gr.Checkbox(label = "Render draft when finished", value = False, elem_id = self.elem_id("render_draft_on_finish"))
                    ue.render_final_on_finish = gr.Checkbox(label = "Render final when finished", value = False, elem_id = self.elem_id("render_final_on_finish"))

                with gr.Row():
                    ue.render_draft = gr.Button(value = "Render draft", elem_id = self.elem_id("render_draft"))
                    ue.render_final = gr.Button(value = "Render final", elem_id = self.elem_id("render_final"))

        ue.render_draft.click(lambda *args: render_video(SimpleNamespace(**{name: arg for name, arg in zip(self.ui_element_names, args)}), False), inputs = list(vars(ue).values()), outputs = [])
        ue.render_final.click(lambda *args: render_video(SimpleNamespace(**{name: arg for name, arg in zip(self.ui_element_names, args)}), True), inputs = list(vars(ue).values()), outputs = [])

        self.ui_element_names = list(vars(ue).keys())

        return list(vars(ue).values())

    def run(self, p, *args):
        uv = SimpleNamespace(**{name: arg for name, arg in zip(self.ui_element_names, args)})

        if not uv.enable:
            return p

        frame_dir = Path(uv.output_dir) / uv.project_subdir
        last_index = get_last_frame_index(frame_dir)

        state.job_count = uv.frame_count

        p.n_iter = 1
        p.batch_size = 1
        p.do_not_save_samples = True
        p.do_not_save_grid = True

        default_p = copy(p)

        if uv.resume and last_index > 0:
            im = Image.open(frame_dir / f"{last_index:05d}.png")
            im.load()
            p.init_images = [im]
            info, _ = images.read_info_from_image(p.init_images[0])
            p.seed = int(info.get("seed", -2)) + 1
        elif uv.generate_first_frame:
            p.init_images = [Image.new("RGB", (p.width, p.height), (128, 128, 128))]
            p.denoising_strength = 1.0

        processing.fix_seed(p)

        if p.init_images:
            color_corrections = [processing.setup_color_correction(p.init_images[0])]
        else:
            color_corrections = None

        # FIXME: Do shit properly, then return; it errors out if the directory doesn't exist
        #with open(frame_dir / "parameters.txt", "w", encoding = "utf-8") as file:
        #    for k, v in vars(uv).items():
        #        file.write(f"{k}: {v}\n")

        for i, frame_index in zip(range(uv.frame_count), count(last_index + 1)):
            state.job = f"Frame {i + 1} / {uv.frame_count}"

            p.init_images = [preprocess_image(p.init_images[0], uv)]
            p.seed = default_p.seed + i

            if opts.img2img_color_correction and color_corrections:
                p.color_corrections = color_corrections

            processed = processing.process_images(p)

            if state.interrupted or state.skipped:
                break

            generated_frame = processed.images[0]
            images.save_image(generated_frame, frame_dir, "", processed.seed, p.prompt, opts.samples_format, info = processed.info, p = p, forced_filename = f"{frame_index:05d}")

            if color_corrections is None:
                color_corrections = [processing.setup_color_correction(generated_frame)]

            p = copy(default_p)
            p.init_images = [generated_frame]

        threads = []

        if uv.render_draft_on_finish:
            thread = Thread(target = render_video, args = (uv, False))
            thread.start()
            threads.append(thread)

        if uv.render_final_on_finish:
            thread = Thread(target = render_video, args = (uv, True))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        return processed
