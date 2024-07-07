from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Any, Optional

from modules import shared as webui_shared
from modules.images import save_image as webui_save_image
from modules.options import Options
from modules.processing import Processed, StableDiffusionProcessing, StableDiffusionProcessingImg2Img, process_images as webui_process_images
from modules.sd_samplers import visible_sampler_names
from modules.shared_state import State

from temporal.backend import Backend, BackendData
from temporal.meta.serializable import SerializableField as Field
from temporal.serialization import BasicObjectSerializer
from temporal.thread_queue import ThreadQueue
from temporal.utils.collection import batched
from temporal.utils.image import PILImage, save_image
from temporal.utils.object import copy_with_overrides
from temporal.webui.controlnet import ControlNetUnitList


# FIXME: To shut up the type checker
opts: Options = getattr(webui_shared, "opts")
state: State = getattr(webui_shared, "state")


class WebUIBackend(Backend):
    def __init__(self) -> None:
        self.processing: StableDiffusionProcessingImg2Img = StableDiffusionProcessingImg2Img()
        self._image_save_queue = ThreadQueue()

    @property
    def samplers(self) -> list[str]:
        return visible_sampler_names()

    @property
    def has_schedulers(self) -> bool:
        return hasattr(StableDiffusionProcessing, "scheduler")

    @property
    def schedulers(self) -> list[str]:
        if self.has_schedulers:
            from modules import sd_schedulers

            return [x.label for x in sd_schedulers.schedulers]
        else:
            return []

    @property
    def is_interrupted(self) -> bool:
        return state.interrupted or state.skipped

    @property
    def is_done_saving(self) -> bool:
        return not self._image_save_queue.busy

    def process_images(self, preview: bool = False, **overrides: Any) -> Optional[list[PILImage]]:
        if not (processed := self._process_inner(
            do_not_save_samples = True,
            do_not_save_grid = True,
            preview = preview,
            **self._unpack_processing_overrides(overrides),
        )) or not processed.images:
            return None

        return processed.images

    def process_batches(self, images: list[tuple[PILImage, int, int]], pixels_per_batch: int = 1048576, preview: bool = False, **overrides: Any) -> Optional[list[list[PILImage]]]:
        first_image, _, _ = images[0]
        pixels_per_image = first_image.width * first_image.height
        batch_size = ceil(pixels_per_batch / pixels_per_image)

        result = defaultdict(list)

        for batch in batched((
            (image_index, image, image_seed)
            for image_index, (image, starting_seed, count) in enumerate(images)
            for image_seed, _ in enumerate(range(count), starting_seed)
        ), batch_size):
            if not (processed := self._process_inner(
                init_images = [image for _, image, _ in batch],
                n_iter = 1,
                batch_size = len(batch),
                seed = [seed for _, _, seed in batch],
                do_not_save_samples = True,
                do_not_save_grid = True,
                preview = preview,
                **self._unpack_processing_overrides(overrides),
            )) or not processed.images:
                return None

            for (image_index, _, _), image in zip(batch, processed.images[:len(batch)]):
                result[image_index].append(image)

        return list(result.values())

    def set_preview_image(self, image: PILImage) -> None:
        state.assign_current_image(image)

    def save_image(self, image: PILImage, output_dir: Path, file_name: Optional[str] = None, archive_mode: bool = False) -> None:
        if file_name and archive_mode:
            self._image_save_queue.enqueue(
                save_image,
                image,
                (output_dir / file_name).with_suffix(".png"),
                archive_mode = True,
            )
        else:
            processed = Processed(self.processing, [image])

            webui_save_image(
                image,
                output_dir,
                "",
                p = self.processing,
                prompt = processed.prompt,
                seed = processed.seed,
                info = processed.info,
                forced_filename = file_name,
                extension = opts.samples_format or "png",
            )

    def _process_inner(self, preview: bool = False, **overrides: Any) -> Optional[Processed]:
        do_set_current_image = State.do_set_current_image

        if not preview:
            State.do_set_current_image = lambda self: None

        try:
            processed = webui_process_images(copy_with_overrides(self.processing, **overrides))
        except:
            State.do_set_current_image = do_set_current_image
            return None

        if state.interrupted or state.skipped:
            State.do_set_current_image = do_set_current_image
            return None

        State.do_set_current_image = do_set_current_image
        return processed

    def _unpack_processing_overrides(self, overrides: dict[str, Any]) -> dict[str, Any]:
        return overrides | {
            "init_images": overrides.pop("images"),
            "sampler_name": overrides.pop("sampler"),
        }


class WebUIBackendData(BackendData):
    options: Options = Field(factory = lambda: copy_with_overrides(opts, data = opts.data.copy()))
    processing: StableDiffusionProcessingImg2Img = Field(factory = StableDiffusionProcessingImg2Img)
    controlnet_units: Optional[ControlNetUnitList] = Field(factory = ControlNetUnitList)

    @property
    def model(self) -> str:
        return self.options.sd_model_checkpoint or "UNDEFINED"

    @property
    def positive_prompt(self) -> str:
        return self.processing.prompt

    @property
    def negative_prompt(self) -> str:
        return self.processing.negative_prompt

    @property
    def images(self) -> list[PILImage]:
        return self.processing.init_images

    @property
    def width(self) -> int:
        return self.processing.width

    @property
    def height(self) -> int:
        return self.processing.height

    @property
    def denoising_strength(self) -> float:
        return self.processing.denoising_strength

    @property
    def seed(self) -> int:
        return self.processing.seed


class _(BasicObjectSerializer[Options], create = False):
    keys = [
        "sd_model_checkpoint",
        "sd_vae",
        "CLIP_stop_at_last_layers",
        "always_discard_next_to_last_sigma",
    ]


class _(BasicObjectSerializer[StableDiffusionProcessingImg2Img], create = False):
    keys = [
        "prompt",
        "negative_prompt",
        "init_images",
        "image_mask",
        "resize_mode",
        "mask_blur_x",
        "mask_blur_y",
        "inpainting_mask_invert",
        "inpainting_fill",
        "inpaint_full_res",
        "inpaint_full_res_padding",
        "sampler_name",
        "steps",
        "refiner_checkpoint",
        "refiner_switch_at",
        "width",
        "height",
        "cfg_scale",
        "denoising_strength",
        "seed",
        "seed_enable_extras",
        "subseed",
        "subseed_strength",
        "seed_resize_from_w",
        "seed_resize_from_h",
    ] + (["scheduler"] if hasattr(StableDiffusionProcessingImg2Img, "scheduler") else [])
