from copy import copy, deepcopy
from itertools import count
from math import ceil
from pathlib import Path

from PIL import Image

from modules import images, processing
from modules.shared import opts, prompt_styles, state

from temporal.fs import clear_directory, ensure_directory_exists, remove_directory
from temporal.image_buffer import ImageBuffer
from temporal.image_preprocessing import PREPROCESSORS, preprocess_image
from temporal.image_utils import average_images, generate_noise_image, save_image
from temporal.metrics import Metrics
from temporal.object_utils import copy_with_overrides
from temporal.session import get_last_frame_index, load_last_frame, load_session, save_session
from temporal.thread_queue import ThreadQueue
from temporal.time_utils import wait_until

image_save_queue = ThreadQueue()

def generate_image(p, ext_params):
    opts_backup = opts.data.copy()

    _apply_prompt_styles(p)

    if not _setup_processing(p):
        return processing.Processed(p, p.init_images)

    image_buffer = ImageBuffer(p.width, p.height, 3, ext_params.frame_merging_frames)
    image_buffer.init(p.init_images[0])

    _apply_relative_params(ext_params, p.denoising_strength)

    images_per_batch = ceil(ext_params.multisampling_samples / ext_params.multisampling_batch_size)

    state.job_count = ext_params.frame_count * images_per_batch

    last_image = p.init_images[0]
    last_info = None

    for i in range(ext_params.frame_count):
        seed = p.seed + i

        if not (processed := _process_image(f"Frame {i + 1} / {ext_params.frame_count}", copy_with_overrides(p,
            init_images = [preprocess_image(last_image, ext_params, seed)],
            n_iter = images_per_batch,
            batch_size = ext_params.multisampling_batch_size,
            seed = seed,
            do_not_save_samples = True,
            do_not_save_grid = True,
        ), ext_params.use_sd)):
            break

        generated_image = average_images(processed.images[:ext_params.multisampling_batch_size * images_per_batch], ext_params.multisampling_algorithm, ext_params.multisampling_easing)
        image_buffer.add(generated_image)
        last_image = image_buffer.average(ext_params.frame_merging_algorithm, ext_params.frame_merging_easing)
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

    image_buffer = ImageBuffer(p.width, p.height, 3, ext_params.frame_merging_frames)
    image_buffer.init(p.init_images[0])

    if ext_params.continue_from_last_frame:
        image_buffer.load(project_dir)

    metrics = Metrics()
    last_index = get_last_frame_index(project_dir)

    if ext_params.metrics_enabled:
        metrics.load(project_dir)

        if last_index == 0:
            metrics.measure(p.init_images[0])

    save_session(p, ext_params, project_dir)

    _apply_relative_params(ext_params, p.denoising_strength)

    images_per_batch = ceil(ext_params.multisampling_samples / ext_params.multisampling_batch_size)

    state.job_count = ext_params.frame_count * images_per_batch

    if last_frame := load_last_frame(project_dir):
        last_image = last_frame
    else:
        last_image = p.init_images[0]

    image_buffer_to_save = deepcopy(image_buffer)

    for i, frame_index in zip(range(ext_params.frame_count), count(last_index + 1)):
        seed = p.seed + frame_index

        if not (processed := _process_image(f"Frame {i + 1} / {ext_params.frame_count}", copy_with_overrides(p,
            init_images = [preprocess_image(last_image, ext_params, seed)],
            n_iter = images_per_batch,
            batch_size = ext_params.multisampling_batch_size,
            seed = seed,
            do_not_save_samples = True,
            do_not_save_grid = True,
        ), ext_params.use_sd)):
            break

        generated_image = average_images(processed.images[:ext_params.multisampling_batch_size * images_per_batch], ext_params.multisampling_algorithm, ext_params.multisampling_easing)
        image_buffer.add(generated_image)
        last_image = image_buffer.average(ext_params.frame_merging_algorithm, ext_params.frame_merging_easing)

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

            image_buffer_to_save = deepcopy(image_buffer)

        if ext_params.metrics_enabled:
            metrics.measure(last_image)
            metrics.save(project_dir)

            if frame_index % ext_params.metrics_save_plots_every_nth_frame == 0:
                metrics.plot(project_dir, save_images = True)

    image_buffer_to_save.save(project_dir)

    wait_until(lambda: not image_save_queue.busy)

    opts.data.update(opts_backup)

    return processing.Processed(p, [last_image])

def _process_image(job_title, p, use_sd = True):
    state.job = job_title

    p = copy(p)

    try:
        if use_sd:
            processed = processing.process_images(p)
        else:
            processed = processing.Processed(p, [p.init_images[0]] * p.n_iter * p.batch_size)
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
        if not (processed := _process_image("Initial image", copy_with_overrides(p,
            init_images = [generate_noise_image((p.width, p.height), p.seed)],
            n_iter = 1,
            batch_size = 1,
            denoising_strength = 1.0,
            do_not_save_samples = True,
            do_not_save_grid = True,
        ))):
            return False

        p.init_images = [processed.images[0]]

    if opts.img2img_color_correction:
        p.color_corrections = [processing.setup_color_correction(p.init_images[0])]

    return True

def _apply_relative_params(ext_params, denoising_strength):
    for key in PREPROCESSORS.keys():
        if getattr(ext_params, f"{key}_amount_relative"):
            setattr(ext_params, f"{key}_amount", getattr(ext_params, f"{key}_amount") * denoising_strength)
