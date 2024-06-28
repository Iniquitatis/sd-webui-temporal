from typing import Optional, Type

import numpy as np
import skimage
from numpy.typing import NDArray

from modules.sd_samplers import visible_sampler_names

from temporal.meta.configurable import BoolParam, Configurable, EnumParam, FloatParam, IntParam
from temporal.meta.serializable import SerializableField as Field
from temporal.project import render_project_video
from temporal.session import Session
from temporal.shared import shared
from temporal.utils.fs import ensure_directory_exists
from temporal.utils.image import NumpyImage, apply_channelwise, ensure_image_dims, match_image, np_to_pil, pil_to_np
from temporal.utils.math import lerp, quantize
from temporal.utils.numpy import average_array, make_eased_weight_array, saturate_array
from temporal.utils.object import copy_with_overrides
from temporal.utils.time import wait_until
from temporal.video_renderer import video_render_queue
from temporal.web_ui import get_schedulers, has_schedulers, image_save_queue, process_images, save_processed_image


PIPELINE_MODULES: dict[str, Type["PipelineModule"]] = {}


class PipelineModule(Configurable, abstract = True):
    store = PIPELINE_MODULES

    icon: str = "\U00002699"

    enabled: bool = Field(False)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        return images

    def finalize(self, images: list[NumpyImage], session: Session) -> None:
        pass


class AveragingModule(PipelineModule):
    id = "averaging"
    name = "Averaging"
    icon = "\U0001f553"

    frames: int = IntParam("Frame count", minimum = 1, step = 1, value = 1, ui_type = "box")
    trimming: float = FloatParam("Trimming", minimum = 0.0, maximum = 0.5, step = 0.01, value = 0.0, ui_type = "slider")
    easing: float = FloatParam("Easing", minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0, ui_type = "slider")
    preference: float = FloatParam("Preference", minimum = -2.0, maximum = 2.0, step = 0.1, value = 0.0, ui_type = "slider")

    buffer: Optional[NDArray[np.float_]] = Field(None)
    last_index: int = Field(0)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if self.buffer is None:
            self.buffer = np.stack([np.repeat(
                ensure_image_dims(image, "RGB", (session.processing.width, session.processing.height))[np.newaxis, ...],
                self.frames,
                axis = 0,
            ) for image in images], 0)
            self.last_index = 0

        for sub, image in zip(self.buffer, images):
            sub[self.last_index] = match_image(image, sub[0])

        self.last_index += 1
        self.last_index %= self.frames

        return [sub[0] if self.frames == 1 else saturate_array(average_array(
            sub,
            axis = 0,
            trim = self.trimming,
            power = self.preference + 1.0,
            weights = np.roll(make_eased_weight_array(self.frames, self.easing), self.last_index),
        )) for sub in self.buffer]


class DetailingModule(PipelineModule):
    id = "detailing"
    name = "Detailing"
    icon = "\U0001f9ec"

    scale: float = FloatParam("Scale", minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0, ui_type = "slider")
    sampler: str = EnumParam("Sampling method", choices = visible_sampler_names(), value = "Euler a", ui_type = "menu")
    if has_schedulers():
        scheduler: str = EnumParam("Schedule type", choices = get_schedulers(), value = "Automatic", ui_type = "menu")
    steps: int = IntParam("Steps", minimum = 1, maximum = 150, step = 1, value = 15, ui_type = "slider")
    denoising_strength: float = FloatParam("Denoising strength", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.2, ui_type = "slider")

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if not (processed_images := process_images(
            copy_with_overrides(session.processing,
                sampler_name = self.sampler,
                steps = self.steps,
                width = quantize(session.processing.width * self.scale, 8),
                height = quantize(session.processing.height * self.scale, 8),
                denoising_strength = self.denoising_strength,
                seed_enable_extras = True,
                seed_resize_from_w = session.processing.seed_resize_from_w or session.processing.width,
                seed_resize_from_h = session.processing.seed_resize_from_h or session.processing.height,
                do_not_save_samples = True,
                do_not_save_grid = True,
                **({"scheduler" : self.scheduler} if hasattr(self, "scheduler") else {})
            ),
            [(np_to_pil(x), seed + i, 1) for i, x in enumerate(images)],
            shared.options.processing.pixels_per_batch,
            shared.previewed_modules[self.id] and not shared.options.live_preview.show_only_finished_images,
        )):
            return None

        return [
            pil_to_np(ensure_image_dims(image_array[0], "RGB", (session.processing.width, session.processing.height)))
            for image_array in processed_images
        ]


class InterpolationModule(PipelineModule):
    id = "interpolation"
    name = "Interpolation"
    icon = "\U0001f553"

    blending: float = FloatParam("Blending", minimum = 0.0, maximum = 1.0, step = 0.001, value = 1.0, ui_type = "slider")
    movement: float = FloatParam("Movement", minimum = 0.0, maximum = 1.0, step = 0.001, value = 1.0, ui_type = "slider")
    radius: int = IntParam("Radius", minimum = 7, maximum = 31, step = 2, value = 15, ui_type = "slider")

    buffer: Optional[NDArray[np.float_]] = Field(None)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if self.buffer is None:
            self.buffer = np.stack([
                ensure_image_dims(image, "RGB", (session.processing.width, session.processing.height))
                for image in images
            ], 0)

        for sub, image in zip(self.buffer, images):
            a = sub
            b = match_image(image, sub)

            if self.movement > 0.0:
                a, b = self._motion_warp(a, b)

            sub[:] = lerp(a, b, self.blending)

        return [sub for sub in self.buffer]

    def _motion_warp(self, base_im: NumpyImage, target_im: NumpyImage) -> tuple[NumpyImage, NumpyImage]:
        def warp(im: NumpyImage, coords: NDArray[np.float_]) -> NumpyImage:
            return apply_channelwise(im, lambda x: skimage.transform.warp(x, coords, mode = "symmetric"))

        height, width = base_im.shape[:2]

        coords = np.array(np.meshgrid(np.arange(height), np.arange(width), indexing = "ij")).astype(np.float_)
        offsets = skimage.registration.optical_flow_ilk(skimage.color.rgb2gray(base_im), skimage.color.rgb2gray(target_im), radius = self.radius)

        return warp(base_im, coords + offsets * -self.movement), warp(target_im, coords + -offsets * (-1.0 + self.movement))


class LimitingModule(PipelineModule):
    id = "limiting"
    name = "Limiting"
    icon = "\U0001f553"

    mode: str = EnumParam("Mode", choices = [("clamp", "Clamp"), ("compress", "Compress")], value = "clamp", ui_type = "menu")
    max_difference: float = FloatParam("Maximum difference", minimum = 0.001, maximum = 1.0, step = 0.001, value = 1.0, ui_type = "slider")

    buffer: Optional[NDArray[np.float_]] = Field(None)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if self.buffer is None:
            self.buffer = np.stack([
                ensure_image_dims(image, "RGB", (session.processing.width, session.processing.height))
                for image in images
            ], 0)

        for sub, image in zip(self.buffer, images):
            a = sub
            b = match_image(image, sub)
            diff = b - a

            if self.mode == "clamp":
                np.clip(diff, -self.max_difference, self.max_difference, out = diff)
            elif self.mode == "compress":
                diff_range = np.abs(diff.max() - diff.min())
                max_diff_range = self.max_difference * 2.0

                if diff_range > max_diff_range:
                    diff *= max_diff_range / diff_range

            sub[:] = saturate_array(a + diff)

        return [sub for sub in self.buffer]


class ProcessingModule(PipelineModule):
    id = "processing"
    name = "Processing"
    icon = "\U0001f9ec"

    samples: int = IntParam("Sample count", minimum = 1, value = 1, ui_type = "box")
    trimming: float = FloatParam("Trimming", minimum = 0.0, maximum = 0.5, step = 0.01, value = 0.0, ui_type = "slider")
    easing: float = FloatParam("Easing", minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0, ui_type = "slider")
    preference: float = FloatParam("Preference", minimum = -2.0, maximum = 2.0, step = 0.1, value = 0.0, ui_type = "slider")

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if not (processed_images := process_images(
            copy_with_overrides(session.processing, do_not_save_samples = True, do_not_save_grid = True),
            [(np_to_pil(x), seed + i, self.samples) for i, x in enumerate(images)],
            shared.options.processing.pixels_per_batch,
            shared.previewed_modules[self.id] and not shared.options.live_preview.show_only_finished_images,
        )):
            return None

        return [
            pil_to_np(image_array[0]) if len(image_array) == 1 else saturate_array(average_array(
                np.stack([pil_to_np(x) for x in image_array]),
                axis = 0,
                trim = self.trimming,
                power = self.preference + 1.0,
                weights = np.flip(make_eased_weight_array(len(image_array), self.easing)),
            ))
            for image_array in processed_images
        ]


class RandomSamplingModule(PipelineModule):
    id = "random_sampling"
    name = "Random sampling"
    icon = "\U0001f553"

    chance: float = FloatParam("Chance", minimum = 0.001, maximum = 1.0, step = 0.001, value = 1.0, ui_type = "slider")

    buffer: Optional[NDArray[np.float_]] = Field(None)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if self.buffer is None:
            self.buffer = np.stack([
                ensure_image_dims(image, "RGB", (session.processing.width, session.processing.height))
                for image in images
            ], 0)

        for i, (sub, image) in enumerate(zip(self.buffer, images)):
            mask = np.random.default_rng(seed + i).random(sub.shape[:2]) <= self.chance

            for j in range(sub.shape[-1]):
                sub[..., j] = np.where(mask, image[..., j], sub[..., j])

        return [sub for sub in self.buffer]


class SavingModule(PipelineModule):
    id = "saving"
    name = "Saving"
    icon = "\U0001f6e0"

    scale: float = FloatParam("Scale", minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0, ui_type = "slider")
    save_every_nth_frame: int = IntParam("Save every N-th frame", minimum = 1, step = 1, value = 1, ui_type = "box")
    save_final: bool = BoolParam("Save final", value = False)
    archive_mode: bool = BoolParam("Archive mode", value = False)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if frame_index % self.save_every_nth_frame != 0:
            return images

        for i, image in enumerate(self._get_scaled_images(images, session)):
            file_name = f"{frame_index:05d}"

            if len(images) > 1:
                file_name += f"-{i + 1:02d}"

            save_processed_image(
                image = np_to_pil(image),
                p = session.processing,
                output_dir = ensure_directory_exists(session.project.path),
                file_name = file_name,
                archive_mode = self.archive_mode,
            )

        return images

    def finalize(self, images: list[NumpyImage], session: Session) -> None:
        if self.save_final:
            for image in self._get_scaled_images(images, session):
                save_processed_image(
                    image = np_to_pil(image),
                    p = session.processing,
                    output_dir = ensure_directory_exists(shared.options.output.output_dir),
                    file_name = None,
                    archive_mode = self.archive_mode,
                )

        wait_until(lambda: not image_save_queue.busy)

    def _get_scaled_images(self, images: list[NumpyImage], session: Session) -> list[NumpyImage]:
        return [ensure_image_dims(x, size = (
            int(quantize(session.processing.width * self.scale, 8)),
            int(quantize(session.processing.height * self.scale, 8)),
        )) for x in images] if self.scale != 1.0 else images


class VideoRenderingModule(PipelineModule):
    id = "video_rendering"
    name = "Video rendering"
    icon = "\U0001f6e0"

    render_draft_every_nth_frame: int = IntParam("Render draft every N-th frame", minimum = 1, step = 1, value = 100, ui_type = "box")
    render_final_every_nth_frame: int = IntParam("Render final every N-th frame", minimum = 1, step = 1, value = 1000, ui_type = "box")
    render_draft_on_finish: bool = BoolParam("Render draft on finish", value = False)
    render_final_on_finish: bool = BoolParam("Render final on finish", value = False)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        for i, _ in enumerate(images, 1):
            if frame_index % self.render_draft_every_nth_frame == 0:
                render_project_video(session.project.path, shared.video_renderer, False, i)

            if frame_index % self.render_final_every_nth_frame == 0:
                render_project_video(session.project.path, shared.video_renderer, True, i)

        return images

    def finalize(self, images: list[NumpyImage], session: Session) -> None:
        for i, _ in enumerate(images, 1):
            if self.render_draft_on_finish:
                render_project_video(session.project.path, shared.video_renderer, False, i)

            if self.render_final_on_finish:
                render_project_video(session.project.path, shared.video_renderer, True, i)

        wait_until(lambda: not video_render_queue.busy)
