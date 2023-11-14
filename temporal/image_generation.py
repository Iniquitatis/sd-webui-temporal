from copy import copy
from itertools import count
from pathlib import Path

import skimage
from PIL import Image

from modules import images, processing
from modules.shared import opts, prompt_styles, state

from temporal.fs import safe_get_directory
from temporal.image_preprocessing import preprocess_image
from temporal.image_utils import generate_noise_image
from temporal.math import lerp
from temporal.metrics import Metrics
from temporal.session import get_last_frame_index, load_session, save_session
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

def generate_project(p, uv):
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
    last_subseed = p.subseed
    #last_subseed_strength = p.subseed_strength

    def noisify(im, seed, level):
        npim = skimage.img_as_float(im)[..., :3]
        noise = skimage.img_as_float(np.random.default_rng(seed).integers(0, 256, size = (im.height, im.width, 3), dtype = "uint8"))
        return lerp(npim, noise, level)

    a = noisify(p.init_images[0], last_seed, uv.drift_noise)
    b = noisify(p.init_images[0], last_seed + 1, uv.drift_noise)

    for i, frame_index in zip(range(uv.frame_count), count(last_index + 1)):
        if not (processed := generate_image(
            f"Frame {i + 1} / {uv.frame_count}",
            p,
            #init_images = [preprocess_image(last_image, uv, last_seed)],
            #init_images = [preprocess_image(p.init_images[0], uv, last_seed)],
            init_images = [preprocess_image(Image.fromarray(skimage.img_as_ubyte(lerp(a, b, (i % uv.drift_factor) / uv.drift_factor))), uv, last_seed)],
            #seed = last_seed,
            #subseed = last_subseed,
            #subseed_strength = last_subseed_strength,
        )):
            return processed

        last_image = processed.images[0]
        # Works! But probably it's slerp that gives those weirdly snappy results
        #last_seed = p.seed + i // uv.drift_factor
        #last_subseed = p.subseed + i // uv.drift_factor
        #last_subseed_strength = (i % uv.drift_factor) / uv.drift_factor

        if i % uv.drift_factor == uv.drift_factor - 1:
            last_seed += 1

            a = b
            b = noisify(p.init_images[0], last_seed, uv.drift_noise)

        if frame_index % uv.save_every_nth_frame == 0:
            if uv.archive_mode:
                image_save_queue.enqueue(
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

    opts.data.update(opts_backup)

    return processed
