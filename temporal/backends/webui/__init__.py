from enum import Enum
from math import floor
from pathlib import Path
from typing import Optional, cast

from modules import processing, shared
from modules.images import resize_image, save_image as webui_save_image
from modules.options import Options
from modules.processing import Processed, StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
from modules.sd_models import checkpoint_tiles
from modules.sd_samplers import visible_sampler_names
from modules.sd_schedulers import schedulers
from modules.sd_vae import vae_dict
from modules.shared_state import State
from modules.styles import StyleDatabase

from temporal.backend import Backend
from temporal.backends.webui.controlnet import ControlNetUnitList, ControlNetUnitWrapper
from temporal.meta.serializable import SerializableField as Field
from temporal.processing_params import ImageToImageParams, TextToImageParams
from temporal.project import Project
from temporal.serialization import BasicObjectSerializer, Serializer
from temporal.thread_queue import ThreadQueue
from temporal.utils.image import NumpyImage, np_to_pil, pil_to_np, save_image
from temporal.utils.object import copy_with_overrides, temporary_patch


# FIXME: To shut up the type checker
opts: Options = getattr(shared, "opts")
prompt_styles: StyleDatabase = getattr(shared, "prompt_styles")
state: State = getattr(shared, "state")


image_save_queue = ThreadQueue()


class WebUITextToImageParams(TextToImageParams):
    options: Options = Field(factory = lambda: copy_with_overrides(opts, data = opts.data.copy()))
    processing: StableDiffusionProcessingTxt2Img = Field(factory = StableDiffusionProcessingTxt2Img)
    controlnet_units: Optional[ControlNetUnitList] = Field(factory = ControlNetUnitList)


class WebUIImageToImageParams(ImageToImageParams):
    options: Options = Field(factory = lambda: copy_with_overrides(opts, data = opts.data.copy()))
    processing: StableDiffusionProcessingImg2Img = Field(factory = StableDiffusionProcessingImg2Img)
    controlnet_units: Optional[ControlNetUnitList] = Field(factory = ControlNetUnitList)


class WebUIBackend(Backend):
    def __init__(self) -> None:
        super().__init__()
        self._last_preview_image = None

    def list_models(self) -> list[str]:
        return [x for x in checkpoint_tiles()]

    def list_vaes(self) -> list[str]:
        return [x for x in vae_dict.keys()]

    def list_upscalers(self) -> list[str]:
        return [x.name for x in shared.sd_upscalers]

    def list_samplers(self) -> list[str]:
        return visible_sampler_names()

    def list_schedulers(self) -> list[str]:
        return [x.label for x in schedulers]

    def text_to_image(self, params: TextToImageParams, preview: bool = False) -> Optional[list[NumpyImage]]:
        params = cast(WebUITextToImageParams, params)

        p = copy_with_overrides(params.processing,
            prompt = params.positive_prompts,
            negative_prompt = params.negative_prompts,
            width = params.width,
            height = params.height,
            sampler_name = params.sampler,
            scheduler = params.scheduler,
            steps = params.steps,
            cfg_scale = params.cfg,
            denoising_strength = params.strength,
            seed = params.seeds,
            seed_enable_extras = True,
            seed_resize_from_w = params.processing.seed_resize_from_w or params.width,
            seed_resize_from_h = params.processing.seed_resize_from_h or params.height,
            n_iter = 1,
            batch_size = max(
                len(params.positive_prompts),
                len(params.negative_prompts),
                len(params.seeds),
            ),
            do_not_save_samples = True,
            do_not_save_grid = True,
        )

        p.prompt = [prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
        p.negative_prompt = [prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in p.negative_prompt]
        p.styles.clear()

        try:
            with (
                temporary_patch(State, "do_set_current_image", State.do_set_current_image if preview else lambda self: None),
                temporary_patch(opts, "save_to_dirs", False),
                temporary_patch(opts, "show_progress_every_n_steps", opts.show_progress_every_n_steps if preview else -1),
                temporary_patch(opts, "sd_model_checkpoint", params.model),
                temporary_patch(opts, "sd_vae", params.vae),
                temporary_patch(opts, "CLIP_stop_at_last_layers", params.clip_skip),
            ):
                processed = processing.process_images(p)
        except:
            return None

        if state.interrupted or state.skipped:
            return None

        return [pil_to_np(x) for x in processed.images]

    def image_to_image(self, params: ImageToImageParams, preview: bool = False) -> Optional[list[NumpyImage]]:
        params = cast(WebUIImageToImageParams, params)

        p = copy_with_overrides(params.processing,
            init_images = [np_to_pil(x) for x in params.images],
            prompt = params.positive_prompts,
            negative_prompt = params.negative_prompts,
            width = params.width,
            height = params.height,
            sampler_name = params.sampler,
            scheduler = params.scheduler,
            steps = params.steps,
            cfg_scale = params.cfg,
            denoising_strength = params.strength,
            seed = params.seeds,
            seed_enable_extras = True,
            seed_resize_from_w = params.processing.seed_resize_from_w or params.width,
            seed_resize_from_h = params.processing.seed_resize_from_h or params.height,
            n_iter = 1,
            batch_size = max(
                len(params.images),
                len(params.positive_prompts),
                len(params.negative_prompts),
                len(params.seeds),
            ),
            do_not_save_samples = True,
            do_not_save_grid = True,
        )

        p.prompt = [prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
        p.negative_prompt = [prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in p.negative_prompt]
        p.styles.clear()

        try:
            with (
                temporary_patch(State, "do_set_current_image", State.do_set_current_image if preview else lambda self: None),
                temporary_patch(opts, "save_to_dirs", False),
                temporary_patch(opts, "show_progress_every_n_steps", opts.show_progress_every_n_steps if preview else -1),
                temporary_patch(opts, "sd_model_checkpoint", params.model),
                temporary_patch(opts, "sd_vae", params.vae),
                temporary_patch(opts, "CLIP_stop_at_last_layers", params.clip_skip),
            ):
                processed = processing.process_images(p)
        except:
            return None

        if state.interrupted or state.skipped:
            return None

        return [pil_to_np(x) for x in processed.images]

    def upscale_image(self, image: NumpyImage, upscaler: str, scale: float) -> Optional[NumpyImage]:
        return pil_to_np(resize_image(0, np_to_pil(image), floor(image.shape[1] * scale), floor(image.shape[0] * scale), upscaler))

    def get_preview(self) -> Optional[NumpyImage]:
        if self._last_preview_image:
            return pil_to_np(self._last_preview_image)

    def set_preview(self, image: Optional[NumpyImage] = None) -> None:
        if image is None:
            if self._last_preview_image is not None:
                state.assign_current_image(self._last_preview_image)

            return

        pil_image = np_to_pil(image)

        state.assign_current_image(pil_image)
        self._last_preview_image = pil_image

    def save_image(self, image: NumpyImage, project: Project, output_dir: Path, file_name: Optional[str] = None, archive_mode: bool = False) -> None:
        pil_image = np_to_pil(image)

        if file_name and archive_mode:
            image_save_queue.enqueue(
                save_image,
                pil_image,
                (output_dir / file_name).with_suffix(".png"),
                archive_mode = True,
            )
        else:
            p = cast(WebUIImageToImageParams, project.parameters).processing
            processed = Processed(p, [pil_image])

            webui_save_image(
                pil_image,
                output_dir,
                "",
                p = p,
                prompt = processed.prompt,
                seed = processed.seed,
                info = processed.info,
                forced_filename = file_name,
                extension = opts.samples_format or "png",
            )

    def are_images_saved(self) -> bool:
        return not image_save_queue.busy

    def interrupt(self) -> None:
        state.interrupt()

    def is_interrupted(self) -> bool:
        return state.interrupted or state.skipped


class _(BasicObjectSerializer[Options], create = False):
    keys = [
        "always_discard_next_to_last_sigma",
    ]


class _(BasicObjectSerializer[StableDiffusionProcessingImg2Img], create = False):
    keys = [
        "image_mask",
        "resize_mode",
        "mask_blur_x",
        "mask_blur_y",
        "inpainting_mask_invert",
        "inpainting_fill",
        "inpaint_full_res",
        "inpaint_full_res_padding",
        "refiner_checkpoint",
        "refiner_switch_at",
        "seed_enable_extras",
        "subseed",
        "subseed_strength",
        "seed_resize_from_w",
        "seed_resize_from_h",
    ]


class _(Serializer[ControlNetUnitWrapper]):
    keys = [
        "image",
        "enabled",
        "low_vram",
        "pixel_perfect",
        "effective_region_mask",
        "module",
        "model",
        "weight",
        "guidance_start",
        "guidance_end",
        "processor_res",
        "threshold_a",
        "threshold_b",
        "control_mode",
        "resize_mode",
    ]

    @classmethod
    def read(cls, obj, ar):
        for key in cls.keys:
            value = ar[key].create()

            if isinstance(object_value := getattr(obj.instance, key), Enum):
                value = type(object_value)(value)

            setattr(obj.instance, key, value)

        return obj

    @classmethod
    def write(cls, obj, ar):
        for key in cls.keys:
            value = getattr(obj.instance, key)

            if isinstance(value, Enum):
                value = value.value

            ar[key].write(value)


class _(Serializer[ControlNetUnitList]):
    @classmethod
    def read(cls, obj, ar):
        for unit, child in zip(obj.units, ar):
            child.read(unit)

        return obj

    @classmethod
    def write(cls, obj, ar):
        for unit in obj.units:
            ar.make_child().write(unit)
