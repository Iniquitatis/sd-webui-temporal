from collections import deque
from copy import copy
from itertools import count
from math import ceil
from pathlib import Path

from PIL import Image

from modules import images, processing
from modules.shared import opts, prompt_styles, state

from temporal.fs import clear_directory, ensure_directory_exists, remove_directory
from temporal.image_preprocessing import PREPROCESSORS, preprocess_image
from temporal.image_utils import ensure_image_dims, generate_noise_image, mean_images, save_image
from temporal.metrics import Metrics
from temporal.session import does_session_exist, get_last_frame_index, load_image_buffer, load_session, save_session, save_image_buffer
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

    project_dir = ensure_directory_exists(Path(ext_params.output_dir) / ext_params.project_subdir)

    if ext_params.start_from_scratch:
        clear_directory(project_dir, "*.png")
        remove_directory(project_dir / "session" / "buffer")
        remove_directory(project_dir / "metrics")

    images_per_batch = ceil(ext_params.image_samples / ext_params.batch_size)
    last_index = get_last_frame_index(project_dir)
    image_buffer = deque(maxlen = ext_params.merged_frames)

    p.prompt = prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
    p.negative_prompt = prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
    p.styles.clear()

    if ext_params.load_session:
        load_session(p, ext_params, project_dir)
        load_image_buffer(image_buffer, project_dir)

    if ext_params.metrics_enabled:
        metrics.load(project_dir)

    processing.fix_seed(p)

    if not p.init_images or not isinstance(p.init_images[0], Image.Image):
        if not (processed := generate_image(
            "Initial image",
            p,
            init_images = [generate_noise_image((p.width, p.height), p.seed)],
            n_iter = 1,
            batch_size = 1,
            denoising_strength = 1.0,
            do_not_save_samples = True,
            do_not_save_grid = True,
        )):
            return processing.Processed(p, p.init_images)

        p.init_images = [processed.images[0]]

    if not image_buffer:
        image_buffer.append(ensure_image_dims(p.init_images[0], "RGB", (p.width, p.height)))

    if ext_params.metrics_enabled and last_index == 0:
        metrics.measure(p.init_images[0])

    if opts.img2img_color_correction:
        p.color_corrections = [processing.setup_color_correction(p.init_images[0])]

    if ext_params.save_session or not does_session_exist(project_dir):
        save_session(p, ext_params, project_dir)

    for key in PREPROCESSORS.keys():
        if getattr(ext_params, f"{key}_amount_relative"):
            setattr(ext_params, f"{key}_amount", getattr(ext_params, f"{key}_amount") * p.denoising_strength)

    state.job_count = ext_params.frame_count * images_per_batch

    for i, frame_index in zip(range(ext_params.frame_count), count(last_index + 1)):
        seed = p.seed + frame_index

        if not (processed := generate_image(
            f"Frame {i + 1} / {ext_params.frame_count}",
            p,
            init_images = [preprocess_image(image_buffer[-1], ext_params, seed)],
            n_iter = images_per_batch,
            batch_size = ext_params.batch_size,
            seed = seed,
            do_not_save_samples = True,
            do_not_save_grid = True,
        )):
            break

        generated_image = mean_images([ensure_image_dims(x, "RGB", (p.width, p.height)) for x in processed.images])
        merged_image = mean_images(image_buffer + deque([generated_image]))
        image_buffer.append(merged_image)

        if frame_index % ext_params.save_every_nth_frame == 0:
            if ext_params.archive_mode:
                image_save_queue.enqueue(
                    save_image,
                    merged_image,
                    project_dir / f"{frame_index:05d}.png",
                    archive_mode = True,
                )
            else:
                images.save_image(
                    merged_image,
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
            metrics.measure(merged_image)
            metrics.save(project_dir)

            if frame_index % ext_params.metrics_save_plots_every_nth_frame == 0:
                metrics.plot(project_dir, save_images = True)

    save_image_buffer(image_buffer, project_dir)

    opts.data.update(opts_backup)

    return processing.Processed(p, [image_buffer[-1]])
