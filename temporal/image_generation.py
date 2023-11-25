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
from temporal.session import get_last_frame_index, load_image_buffer, load_last_frame, load_session, save_image_buffer, save_session
from temporal.thread_queue import ThreadQueue

image_save_queue = ThreadQueue()

def generate_image(p, ext_params):
    opts_backup = opts.data.copy()

    _apply_prompt_styles(p)

    if not _setup_processing(p):
        return processing.Processed(p, p.init_images)

    image_buffer = deque([ensure_image_dims(p.init_images[0], "RGB", (p.width, p.height))], maxlen = ext_params.merged_frames)
    _pad_image_buffer(image_buffer)

    _apply_relative_params(ext_params, p.denoising_strength)

    images_per_batch = ceil(ext_params.image_samples / ext_params.batch_size)

    state.job_count = ext_params.frame_count * images_per_batch

    last_image = p.init_images[0]
    last_info = None

    for i in range(ext_params.frame_count):
        seed = p.seed + i

        if not (processed := _process_image(
            f"Frame {i + 1} / {ext_params.frame_count}",
            p,
            init_images = [preprocess_image(last_image, ext_params, seed)],
            n_iter = images_per_batch,
            batch_size = ext_params.batch_size,
            seed = seed,
            do_not_save_samples = True,
            do_not_save_grid = True,
        )):
            break

        generated_image = mean_images(processed.images)
        image_buffer.append(generated_image)
        last_image = mean_images(image_buffer)
        last_info = processed.info

    images.save_image(
        last_image,
        ext_params.output_dir,
        "",
        p = p,
        prompt = p.prompt,
        seed = p.seed,
        info = last_info,
        extension = opts.samples_format,
    )

    opts.data.update(opts_backup)

    return processing.Processed(p, [last_image])

def generate_sequence(p, ext_params):
    opts_backup = opts.data.copy()

    project_dir = ensure_directory_exists(Path(ext_params.output_dir) / ext_params.project_subdir)

    if not ext_params.continue_from_last_frame:
        clear_directory(project_dir, "*.png")
        remove_directory(project_dir / "session" / "buffer")
        remove_directory(project_dir / "metrics")

    _apply_prompt_styles(p)

    if ext_params.load_parameters:
        load_session(p, ext_params, project_dir)

    if not _setup_processing(p):
        return processing.Processed(p, p.init_images)

    image_buffer = deque(maxlen = ext_params.merged_frames)

    if ext_params.continue_from_last_frame:
        load_image_buffer(image_buffer, project_dir)

    if not image_buffer:
        image_buffer.append(ensure_image_dims(p.init_images[0], "RGB", (p.width, p.height)))

    _pad_image_buffer(image_buffer)

    metrics = Metrics()
    last_index = get_last_frame_index(project_dir)

    if ext_params.metrics_enabled:
        metrics.load(project_dir)

        if last_index == 0:
            metrics.measure(p.init_images[0])

    save_session(p, ext_params, project_dir)

    _apply_relative_params(ext_params, p.denoising_strength)

    images_per_batch = ceil(ext_params.image_samples / ext_params.batch_size)

    state.job_count = ext_params.frame_count * images_per_batch

    if last_frame := load_last_frame(project_dir):
        last_image = last_frame
    else:
        last_image = p.init_images[0]

    image_buffer_to_save = image_buffer.copy()

    for i, frame_index in zip(range(ext_params.frame_count), count(last_index + 1)):
        seed = p.seed + frame_index

        if not (processed := _process_image(
            f"Frame {i + 1} / {ext_params.frame_count}",
            p,
            init_images = [preprocess_image(last_image, ext_params, seed)],
            n_iter = images_per_batch,
            batch_size = ext_params.batch_size,
            seed = seed,
            do_not_save_samples = True,
            do_not_save_grid = True,
        )):
            break

        generated_image = mean_images(processed.images)
        image_buffer.append(generated_image)
        last_image = mean_images(image_buffer)

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
                    p = p,
                    prompt = p.prompt,
                    seed = processed.seed,
                    info = processed.info,
                    forced_filename = f"{frame_index:05d}",
                    extension = opts.samples_format,
                )

            image_buffer_to_save = image_buffer.copy()

        if ext_params.metrics_enabled:
            metrics.measure(last_image)
            metrics.save(project_dir)

            if frame_index % ext_params.metrics_save_plots_every_nth_frame == 0:
                metrics.plot(project_dir, save_images = True)

    save_image_buffer(image_buffer_to_save, project_dir)

    opts.data.update(opts_backup)

    return processing.Processed(p, [last_image])

def generate_prompt_travel(p, ext_params):
    opts_backup = opts.data.copy()

    project_dir = ensure_directory_exists(Path(ext_params.output_dir) / ext_params.project_subdir)

    if not ext_params.continue_from_last_frame:
        clear_directory(project_dir, "*.png")

    _apply_prompt_styles(p)

    if ext_params.load_parameters:
        load_session(p, ext_params, project_dir)

    if not _setup_processing(p):
        return processing.Processed(p, p.init_images)

    save_session(p, ext_params, project_dir)

    _apply_relative_params(ext_params, p.denoising_strength)

    images_per_batch = ceil(ext_params.image_samples / ext_params.batch_size)

    state.job_count = ext_params.frame_count * images_per_batch

    last_image = None

    for i in range(get_last_frame_index(project_dir), ext_params.frame_count):
        factor = i * ext_params.prompt_travel_rate
        prompt_a = f"({ext_params.prompt_travel_prompt_a}:{1.0 - factor})"
        prompt_b = f"({ext_params.prompt_travel_prompt_b}:{factor})"

        if not (processed := _process_image(
            f"Frame {i + 1} / {ext_params.frame_count}",
            p,
            prompt = f"{p.prompt}, {prompt_a}, {prompt_b}",
            init_images = [preprocess_image(p.init_images[0], ext_params, p.seed)],
            n_iter = images_per_batch,
            batch_size = ext_params.batch_size,
            do_not_save_samples = True,
            do_not_save_grid = True,
        )):
            break

        last_image = mean_images(processed.images)

        if i % ext_params.save_every_nth_frame == 0:
            if ext_params.archive_mode:
                image_save_queue.enqueue(
                    save_image,
                    last_image,
                    project_dir / f"{i:05d}.png",
                    archive_mode = True,
                )
            else:
                images.save_image(
                    last_image,
                    project_dir,
                    "",
                    p = p,
                    prompt = p.prompt,
                    seed = processed.seed,
                    info = processed.info,
                    forced_filename = f"{i:05d}",
                    extension = opts.samples_format,
                )

    opts.data.update(opts_backup)

    return processing.Processed(p, [last_image or p.init_images[0]])

def _process_image(job_title, p, **p_overrides):
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

def _apply_prompt_styles(p):
    p.prompt = prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
    p.negative_prompt = prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
    p.styles.clear()

def _setup_processing(p):
    processing.fix_seed(p)

    if not p.init_images or not isinstance(p.init_images[0], Image.Image):
        if not (processed := _process_image(
            "Initial image",
            p,
            init_images = [generate_noise_image((p.width, p.height), p.seed)],
            n_iter = 1,
            batch_size = 1,
            denoising_strength = 1.0,
            do_not_save_samples = True,
            do_not_save_grid = True,
        )):
            return False

        p.init_images = [processed.images[0]]

    if opts.img2img_color_correction:
        p.color_corrections = [processing.setup_color_correction(p.init_images[0])]

    return True

def _apply_relative_params(ext_params, denoising_strength):
    for key in PREPROCESSORS.keys():
        if getattr(ext_params, f"{key}_amount_relative"):
            setattr(ext_params, f"{key}_amount", getattr(ext_params, f"{key}_amount") * denoising_strength)

def _pad_image_buffer(image_buffer):
    if not image_buffer:
        return

    first_image = image_buffer[0]

    while len(image_buffer) < (image_buffer.maxlen or 0):
        image_buffer.insert(0, first_image)
