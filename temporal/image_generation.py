from copy import copy
from itertools import count
from math import ceil
from pathlib import Path

from PIL import Image

from modules import images, processing
from modules.shared import opts, prompt_styles, state

from temporal.fs import safe_get_directory
from temporal.image_preprocessing import PREPROCESSORS, preprocess_image
from temporal.image_utils import generate_noise_image, mean_images, save_image
from temporal.metrics import Metrics
from temporal.session import does_session_exist, get_last_frame_index, load_session, save_session
from temporal.thread_queue import ThreadQueue

image_save_queue = ThreadQueue()

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

def generate_project(p, ext_params):
    metrics = Metrics()

    opts_backup = opts.data.copy()

    project_dir = safe_get_directory(Path(ext_params.output_dir) / ext_params.project_subdir)

    if ext_params.start_from_scratch:
        for path in project_dir.glob("*.png"):
            path.unlink()

        metrics.clear(project_dir)

    last_index = get_last_frame_index(project_dir)

    p.prompt = prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
    p.negative_prompt = prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
    p.styles.clear()

    if ext_params.load_session:
        load_session(p, ext_params, project_dir)

    if ext_params.metrics_enabled:
        metrics.load(project_dir)

    p.n_iter = ceil(ext_params.image_samples / ext_params.batch_size)
    p.batch_size = ext_params.batch_size
    p.do_not_save_samples = True
    p.do_not_save_grid = True
    processing.fix_seed(p)

    if not p.init_images or not isinstance(p.init_images[0], Image.Image):
        if processed := generate_image(
            "Initial image",
            p,
            init_images = [generate_noise_image((p.width, p.height), p.seed)],
            n_iter = 1,
            batch_size = 1,
            denoising_strength = 1.0,
        ):
            p.init_images = [processed.images[0]]
            p.seed += 1
        else:
            return processing.Processed(p, p.init_images)

    if ext_params.metrics_enabled and last_index == 0:
        metrics.measure(p.init_images[0])

    if opts.img2img_color_correction:
        p.color_corrections = [processing.setup_color_correction(p.init_images[0])]

    if ext_params.save_session or not does_session_exist(project_dir):
        save_session(p, ext_params, project_dir)

    for key in PREPROCESSORS.keys():
        if getattr(ext_params, f"{key}_amount_relative"):
            setattr(ext_params, f"{key}_amount", getattr(ext_params, f"{key}_amount") * p.denoising_strength)

    state.job_count = ext_params.frame_count * p.n_iter

    last_image = p.init_images[0]
    last_seed = p.seed

    for i, frame_index in zip(range(ext_params.frame_count), count(last_index + 1)):
        if not (processed := generate_image(
            f"Frame {i + 1} / {ext_params.frame_count}",
            p,
            init_images = [preprocess_image(last_image, ext_params, last_seed)],
            seed = last_seed,
        )):
            processed = processing.Processed(p, [last_image])
            break

        last_image = mean_images(processed.images)
        last_seed += 1

        if frame_index % ext_params.save_every_nth_frame == 0:
            if ext_params.archive_mode:
                image_save_queue.enqueue(
                    save_image,
                    last_image,
                    project_dir / f"{frame_index:05d}.png",
                    archive_mode = True,
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

        if ext_params.metrics_enabled:
            metrics.measure(last_image)
            metrics.save(project_dir)

            if frame_index % ext_params.metrics_save_plots_every_nth_frame == 0:
                metrics.plot(project_dir, save_images = True)

    opts.data.update(opts_backup)

    return processed
