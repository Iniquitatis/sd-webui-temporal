from collections import defaultdict
from copy import copy
from math import ceil
from pathlib import Path
from typing import Optional

from modules import images, processing, shared
from modules.options import Options
from modules.processing import Processed, StableDiffusionProcessing, StableDiffusionProcessingImg2Img
from modules.shared_state import State

from temporal.thread_queue import ThreadQueue
from temporal.utils.collection import batched
from temporal.utils.image import PILImage, save_image
from temporal.utils.object import copy_with_overrides


# FIXME: To shut up the type checker
opts: Options = getattr(shared, "opts")
state: State = getattr(shared, "state")


image_save_queue = ThreadQueue()


def process_image(p: StableDiffusionProcessing, preview: bool = False) -> Optional[Processed]:
    p = copy(p)

    do_set_current_image = State.do_set_current_image

    if not preview:
        State.do_set_current_image = lambda self: None

    try:
        processed = processing.process_images(p)
    except:
        State.do_set_current_image = do_set_current_image
        return None

    if state.interrupted or state.skipped:
        State.do_set_current_image = do_set_current_image
        return None

    State.do_set_current_image = do_set_current_image
    return processed


def process_images(p: StableDiffusionProcessingImg2Img, images: list[tuple[PILImage, int, int]], pixels_per_batch: int = 1048576, preview: bool = False) -> Optional[list[list[PILImage]]]:
    first_image, _, _ = images[0]
    pixels_per_image = first_image.width * first_image.height
    batch_size = ceil(pixels_per_batch / pixels_per_image)

    result = defaultdict(list)

    for batch in batched((
        (image_index, image, image_seed)
        for image_index, (image, starting_seed, count) in enumerate(images)
        for image_seed, _ in enumerate(range(count), starting_seed)
    ), batch_size):
        if not (processed := process_image(copy_with_overrides(p,
            init_images = [image for _, image, _ in batch],
            n_iter = 1,
            batch_size = len(batch),
            seed = [seed for _, _, seed in batch],
        ), preview)) or not processed.images:
            return None

        for (image_index, _, _), image in zip(batch, processed.images[:len(batch)]):
            result[image_index].append(image)

    return list(result.values())


def save_processed_image(image: PILImage, p: StableDiffusionProcessing, output_dir: Path, file_name: Optional[str] = None, archive_mode: bool = False) -> None:
    if file_name and archive_mode:
        image_save_queue.enqueue(
            save_image,
            image,
            (output_dir / file_name).with_suffix(".png"),
            archive_mode = True,
        )
    else:
        processed = Processed(p, [image])

        images.save_image(
            image,
            output_dir,
            "",
            p = p,
            prompt = processed.prompt,
            seed = processed.seed,
            info = processed.info,
            forced_filename = file_name,
            extension = opts.samples_format or "png",
        )
