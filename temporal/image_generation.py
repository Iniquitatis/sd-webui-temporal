from copy import copy, deepcopy
from itertools import count
from math import ceil
from pathlib import Path

from PIL import Image

from modules import images, processing
from modules.shared import opts, prompt_styles, state

from temporal.fs import clear_directory, ensure_directory_exists, remove_directory
from temporal.func_utils import make_func_registerer
from temporal.image_buffer import ImageBuffer
from temporal.image_preprocessing import PREPROCESSORS, preprocess_image
from temporal.image_utils import average_images, ensure_image_dims, generate_value_noise_image, save_image
from temporal.metrics import Metrics
from temporal.object_utils import copy_with_overrides
from temporal.session import get_last_frame_index, load_last_frame, load_session, save_session
from temporal.thread_queue import ThreadQueue
from temporal.time_utils import wait_until

GENERATION_MODES, generation_mode = make_func_registerer(name = "")

image_save_queue = ThreadQueue()

@generation_mode("image", "Image")
def _(p, ext_params):
    opts_backup = opts.data.copy()

    _apply_prompt_styles(p)

    if not _setup_processing(p, ext_params):
        return processing.Processed(p, p.init_images)

    image_buffer = _make_image_buffer(p, ext_params)

    _apply_relative_params(ext_params, p.denoising_strength)

    last_processed = processing.Processed(p, [p.init_images[0]])

    for i in range(ext_params.frame_count):
        if not (processed := _process_iteration(
            p = p,
            ext_params = ext_params,
            image_buffer = image_buffer,
            image = last_processed.images[0],
            i = i,
            frame_index = i + 1,
        )):
            break

        last_processed = processed

    _save_processed_image(
        p = p,
        processed = last_processed,
        output_dir = ext_params.output_dir,
    )

    opts.data.update(opts_backup)

    return last_processed

@generation_mode("sequence", "Sequence")
def _(p, ext_params):
    opts_backup = opts.data.copy()

    opts.save_to_dirs = False

    project_dir = ensure_directory_exists(Path(ext_params.output_dir) / ext_params.project_subdir)

    if not ext_params.continue_from_last_frame:
        clear_directory(project_dir, "*.png")
        remove_directory(project_dir / "session" / "buffer")
        remove_directory(project_dir / "metrics")

    _apply_prompt_styles(p)

    if ext_params.load_parameters:
        load_session(p, ext_params, project_dir)

    if not _setup_processing(p, ext_params):
        return processing.Processed(p, p.init_images)

    image_buffer = _make_image_buffer(p, ext_params)

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

    if last_frame := load_last_frame(project_dir):
        last_processed = processing.Processed(p, [last_frame])
    else:
        last_processed = processing.Processed(p, [p.init_images[0]])

    image_buffer_to_save = deepcopy(image_buffer)

    for i, frame_index in zip(range(ext_params.frame_count), count(last_index + 1)):
        if not (processed := _process_iteration(
            p = p,
            ext_params = ext_params,
            image_buffer = image_buffer,
            image = last_processed.images[0],
            i = i,
            frame_index = frame_index,
        )):
            break

        last_processed = processed

        if frame_index % ext_params.save_every_nth_frame == 0:
            _save_processed_image(
                p = p,
                processed = last_processed,
                output_dir = project_dir,
                file_name = f"{frame_index:05d}",
                archive_mode = ext_params.archive_mode,
            )

            image_buffer_to_save = deepcopy(image_buffer)

        if ext_params.metrics_enabled:
            metrics.measure(last_processed.images[0])
            metrics.save(project_dir)

            if frame_index % ext_params.metrics_save_plots_every_nth_frame == 0:
                metrics.plot(project_dir, save_images = True)

    image_buffer_to_save.save(project_dir)

    wait_until(lambda: not image_save_queue.busy)

    opts.data.update(opts_backup)

    return last_processed

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

    if not use_sd:
        state.nextjob()
        state.assign_current_image(processed.images[0])

    return processed

def _apply_prompt_styles(p):
    p.prompt = prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
    p.negative_prompt = prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
    p.styles.clear()

def _setup_processing(p, ext_params):
    processing.fix_seed(p)

    if not p.init_images or not isinstance(p.init_images[0], Image.Image):
        if not (processed := _process_image("Initial image", copy_with_overrides(p,
            init_images = [generate_value_noise_image(
                (p.width, p.height),
                3,
                ext_params.initial_noise_scale,
                ext_params.initial_noise_octaves,
                ext_params.initial_noise_lacunarity,
                ext_params.initial_noise_persistence,
                p.seed,
            )],
            n_iter = 1,
            batch_size = 1,
            denoising_strength = 1.0 - ext_params.initial_noise_factor,
            do_not_save_samples = True,
            do_not_save_grid = True,
        ), ext_params.initial_noise_factor < 1.0)) or not processed.images:
            return False

        p.init_images = [processed.images[0]]

    if opts.img2img_color_correction:
        p.color_corrections = [processing.setup_color_correction(p.init_images[0])]

    return True

def _make_image_buffer(p, ext_params):
    width = p.width
    height = p.height

    if ext_params.detailing_scale_buffer:
        width *= ext_params.detailing_scale
        height *= ext_params.detailing_scale

    buffer = ImageBuffer(width, height, 3, ext_params.frame_merging_frames)
    buffer.init(p.init_images[0])

    return buffer

def _apply_relative_params(ext_params, denoising_strength):
    for key in PREPROCESSORS.keys():
        if getattr(ext_params, f"{key}_amount_relative"):
            setattr(ext_params, f"{key}_amount", getattr(ext_params, f"{key}_amount") * denoising_strength)

def _process_iteration(p, ext_params, image_buffer, image, i, frame_index):
    batch_count = ceil(ext_params.multisampling_samples / ext_params.multisampling_batch_size)
    total_sample_count = ext_params.multisampling_batch_size * batch_count

    state.job_count = ext_params.frame_count * batch_count

    if ext_params.detailing_enabled:
        state.job_count *= 2

    seed = p.seed + frame_index

    if not (processed := _process_image(f"Frame {i + 1} / {ext_params.frame_count}", copy_with_overrides(p,
        init_images = [preprocess_image(image, ext_params, seed)],
        n_iter = batch_count,
        batch_size = ext_params.multisampling_batch_size,
        seed = seed,
        do_not_save_samples = True,
        do_not_save_grid = True,
    ), ext_params.use_sd)) or not processed.images:
        return None

    samples = processed.images[:total_sample_count]

    if ext_params.detailing_enabled:
        detailed_samples = []

        for j in range(batch_count):
            offset_from = j * ext_params.multisampling_batch_size
            offset_to = (j + 1) * ext_params.multisampling_batch_size

            if not (detailed := _process_image("Detailing", copy_with_overrides(p,
                init_images = samples[offset_from:offset_to],
                sampler_name = ext_params.detailing_sampler,
                steps = ext_params.detailing_steps,
                width = p.width * ext_params.detailing_scale,
                height = p.height * ext_params.detailing_scale,
                n_iter = 1,
                batch_size = ext_params.multisampling_batch_size,
                denoising_strength = ext_params.detailing_denoising_strength,
                seed = seed + total_sample_count + offset_from,
                seed_enable_extras = True,
                seed_resize_from_w = p.seed_resize_from_w or p.width,
                seed_resize_from_h = p.seed_resize_from_h or p.height,
                do_not_save_samples = True,
                do_not_save_grid = True,
            ), ext_params.use_sd)) or not detailed.images:
                return None

            detailed_samples.extend(
                ensure_image_dims(x, x.mode, (image_buffer.width, image_buffer.height))
                for x in detailed.images[:ext_params.multisampling_batch_size]
            )

        samples = detailed_samples

    multisampled_image = average_images(samples, ext_params.multisampling_trimming, ext_params.multisampling_easing, ext_params.multisampling_preference)
    image_buffer.add(multisampled_image)
    merged_image = image_buffer.average(ext_params.frame_merging_trimming, ext_params.frame_merging_easing, ext_params.frame_merging_preference)
    merged_image = ensure_image_dims(merged_image, merged_image.mode, (p.width, p.height))

    return copy_with_overrides(processed, images = [merged_image])

def _save_processed_image(p, processed, output_dir, file_name = None, archive_mode = False):
    if archive_mode:
        image_save_queue.enqueue(
            save_image,
            processed.images[0],
            (output_dir / file_name).with_suffix(".png"),
            archive_mode = True,
        )
    else:
        images.save_image(
            processed.images[0],
            output_dir,
            "",
            p = p,
            prompt = processed.prompt,
            seed = processed.seed,
            info = processed.info,
            forced_filename = file_name,
            extension = opts.samples_format,
        )
