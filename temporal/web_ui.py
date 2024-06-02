from copy import copy
from pathlib import Path
from typing import Optional

from modules import images, processing, shared
from modules.options import Options
from modules.processing import Processed, StableDiffusionProcessing
from modules.shared_state import State

from temporal.thread_queue import ThreadQueue
from temporal.utils.image import PILImage, save_image


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


def save_processed_image(image: PILImage, p: StableDiffusionProcessing, processed: Processed, output_dir: Path, file_name: Optional[str] = None, archive_mode: bool = False) -> None:
    if file_name and archive_mode:
        image_save_queue.enqueue(
            save_image,
            image,
            (output_dir / file_name).with_suffix(".png"),
            archive_mode = True,
        )
    else:
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


def set_preview_image(image: PILImage) -> None:
    global _last_preview_image

    if image is not _last_preview_image:
        state.assign_current_image(image)
        _last_preview_image = image


_last_preview_image: Optional[PILImage] = None
