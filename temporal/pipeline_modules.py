from math import ceil
from typing import Optional, Type

import gradio as gr
import numpy as np
from numpy.typing import NDArray

from modules.processing import Processed
from modules.sd_samplers import visible_sampler_names

from temporal.image_buffer import ImageBuffer
from temporal.meta.configurable import Configurable, ui_param
from temporal.meta.serializable import field
from temporal.metrics import Metrics
from temporal.project import render_project_video
from temporal.session import Session
from temporal.utils.fs import ensure_directory_exists
from temporal.utils.image import PILImage, average_images, ensure_image_dims, match_image, np_to_pil, pil_to_np
from temporal.utils.math import lerp, quantize
from temporal.utils.numpy import saturate_array
from temporal.utils.object import copy_with_overrides, get_with_overrides
from temporal.utils.time import wait_until
from temporal.video_renderer import video_render_queue
from temporal.web_ui import image_save_queue, process_image, save_processed_image


PIPELINE_MODULES: dict[str, Type["PipelineModule"]] = {}


class PipelineModule(Configurable, abstract = True):
    store = PIPELINE_MODULES

    enabled: bool = field(False)
    preview: bool = field(True)

    def forward(self, session: Session, processed: Processed, frame_index: int, frame_count: int, seed: int) -> Optional[Processed]:
        return processed

    def finalize(self, session: Session, processed: Processed) -> None:
        pass


class DampeningModule(PipelineModule):
    id = "dampening"
    name = "Dampening"

    rate: float = ui_param("Rate", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.001, value = 1.0)

    buffer: NDArray[np.float_] = field(factory = lambda: np.empty((0, 0)))

    def forward(self, session: Session, processed: Processed, frame_index: int, frame_count: int, seed: int) -> Optional[Processed]:
        if self.buffer.shape[0] == 0:
            self.buffer = pil_to_np(ensure_image_dims(processed.images[0], "RGB", (session.processing.width, session.processing.height)))

        self.buffer[:] = lerp(
            self.buffer,
            pil_to_np(match_image(processed.images[0], self.buffer)),
            self.rate,
        )

        return get_with_overrides(processed,
            images = [np_to_pil(self.buffer)],
        )


class DetailingModule(PipelineModule):
    id = "detailing"
    name = "Detailing"

    scale: float = ui_param("Scale", gr.Slider, minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0)
    sampler: str = ui_param("Sampling method", gr.Dropdown, choices = visible_sampler_names(), value = "Euler a")
    steps: int = ui_param("Steps", gr.Slider, precision = 0, minimum = 1, maximum = 150, step = 1, value = 15)
    denoising_strength: float = ui_param("Denoising strength", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.2)

    def forward(self, session: Session, processed: Processed, frame_index: int, frame_count: int, seed: int) -> Optional[Processed]:
        if not (new_processed := process_image(copy_with_overrides(session.processing,
            init_images = processed.images,
            sampler_name = self.sampler,
            steps = self.steps,
            width = quantize(session.processing.width * self.scale, 8),
            height = quantize(session.processing.height * self.scale, 8),
            n_iter = 1,
            batch_size = 1,
            denoising_strength = self.denoising_strength,
            seed_enable_extras = True,
            seed_resize_from_w = session.processing.seed_resize_from_w or session.processing.width,
            seed_resize_from_h = session.processing.seed_resize_from_h or session.processing.height,
            do_not_save_samples = True,
            do_not_save_grid = True,
        ), self.preview)) or not new_processed.images:
            return None

        return get_with_overrides(new_processed,
            images = [ensure_image_dims(x, size = (session.processing.width, session.processing.height)) for x in new_processed.images],
        )


class FrameMergingModule(PipelineModule):
    id = "frame_merging"
    name = "Frame merging"

    buffer_scale: float = ui_param("Buffer scale", gr.Slider, minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0)
    frames: int = ui_param("Frame count", gr.Number, precision = 0, minimum = 1, step = 1, value = 1)
    trimming: float = ui_param("Trimming", gr.Slider, minimum = 0.0, maximum = 0.5, step = 0.01, value = 0.0)
    easing: float = ui_param("Easing", gr.Slider, minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0)
    preference: float = ui_param("Preference", gr.Slider, minimum = -2.0, maximum = 2.0, step = 0.1, value = 0.0)

    buffer: ImageBuffer = field(factory = ImageBuffer)

    def forward(self, session: Session, processed: Processed, frame_index: int, frame_count: int, seed: int) -> Optional[Processed]:
        if self.buffer.array.shape[0] == 0:
            self.buffer.init(
                int(quantize(session.processing.width * self.buffer_scale, 8)),
                int(quantize(session.processing.height * self.buffer_scale, 8)),
            3, self.frames, processed.images[0])

        self.buffer.add(processed.images[0])
        return get_with_overrides(processed,
            images = [self.buffer.average(self.trimming, self.easing, self.preference)],
        )


class ImageFilteringModule(PipelineModule):
    id = "image_filtering"
    name = "Image filtering"

    def forward(self, session: Session, processed: Processed, frame_index: int, frame_count: int, seed: int) -> Optional[Processed]:
        return get_with_overrides(processed,
            images = [session.image_filterer.filter_image(x, session.processing.denoising_strength, seed) for x in processed.images],
        )


class LimitingModule(PipelineModule):
    id = "limiting"
    name = "Limiting"

    mode: str = ui_param("Mode", gr.Dropdown, choices = ["clamp", "compress"], value = "clamp")
    max_difference: float = ui_param("Maximum difference", gr.Slider, minimum = 0.001, maximum = 1.0, step = 0.001, value = 1.0)

    buffer: NDArray[np.float_] = field(factory = lambda: np.empty((0, 0)))

    def forward(self, session: Session, processed: Processed, frame_index: int, frame_count: int, seed: int) -> Optional[Processed]:
        if self.buffer.shape[0] == 0:
            self.buffer = pil_to_np(ensure_image_dims(processed.images[0], "RGB", (session.processing.width, session.processing.height)))

        a = self.buffer
        b = pil_to_np(match_image(processed.images[0], self.buffer))
        diff = b - a

        if self.mode == "clamp":
            np.clip(diff, -self.max_difference, self.max_difference, out = diff)
        elif self.mode == "compress":
            diff_range = np.abs(diff.max() - diff.min())
            max_diff_range = self.max_difference * 2.0

            if diff_range > max_diff_range:
                diff *= max_diff_range / diff_range

        self.buffer[:] = saturate_array(a + diff)

        return get_with_overrides(processed,
            images = [np_to_pil(self.buffer)],
        )


class MeasuringModule(PipelineModule):
    id = "measuring"
    name = "Measuring"

    plot_every_nth_frame: int = ui_param("Plot every N-th frame", gr.Number, precision = 0, minimum = 1, step = 1, value = 10)

    metrics: Metrics = field(factory = Metrics)

    def forward(self, session: Session, processed: Processed, frame_index: int, frame_count: int, seed: int) -> Optional[Processed]:
        self.metrics.measure(processed.images[0])

        if frame_index % self.plot_every_nth_frame == 0:
            self.metrics.plot_to_directory(ensure_directory_exists(session.output.output_dir / session.output.project_subdir / "metrics"))

        return processed


class ProcessingModule(PipelineModule):
    id = "processing"
    name = "Processing"

    samples: int = ui_param("Sample count", gr.Number, precision = 0, minimum = 1, value = 1)
    batch_size: int = ui_param("Batch size", gr.Number, precision = 0, minimum = 1, value = 1)
    trimming: float = ui_param("Trimming", gr.Slider, minimum = 0.0, maximum = 0.5, step = 0.01, value = 0.0)
    easing: float = ui_param("Easing", gr.Slider, minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0)
    preference: float = ui_param("Preference", gr.Slider, minimum = -2.0, maximum = 2.0, step = 0.1, value = 0.0)

    def forward(self, session: Session, processed: Processed, frame_index: int, frame_count: int, seed: int) -> Optional[Processed]:
        batch_count = ceil(self.samples / self.batch_size)

        if not (new_processed := process_image(copy_with_overrides(session.processing,
            init_images = processed.images,
            n_iter = batch_count,
            batch_size = self.batch_size,
            seed = seed,
            do_not_save_samples = True,
            do_not_save_grid = True,
        ), self.preview)) or not new_processed.images:
            return None

        return get_with_overrides(new_processed,
            images = [average_images(new_processed.images[:self.batch_size * batch_count], self.trimming, self.easing, self.preference)],
        )


class SavingModule(PipelineModule):
    id = "saving"
    name = "Saving"

    scale: float = ui_param("Scale", gr.Slider, minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0)
    save_every_nth_frame: int = ui_param("Save every N-th frame", gr.Number, precision = 0, minimum = 1, step = 1, value = 1)
    save_final: bool = ui_param("Save final", gr.Checkbox, value = False)
    archive_mode: bool = ui_param("Archive mode", gr.Checkbox, value = False)

    def forward(self, session: Session, processed: Processed, frame_index: int, frame_count: int, seed: int) -> Optional[Processed]:
        if frame_index % self.save_every_nth_frame != 0:
            return processed

        images = self._get_scaled_images(session, processed)

        for i, image in enumerate(images):
            file_name = f"{frame_index:05d}"

            if len(images) > 1:
                file_name += f"-{i}"

            save_processed_image(
                image = image,
                p = session.processing,
                processed = processed,
                output_dir = ensure_directory_exists(session.output.output_dir / session.output.project_subdir),
                file_name = file_name,
                archive_mode = self.archive_mode,
            )

        return processed

    def finalize(self, session: Session, processed: Processed) -> None:
        if self.save_final:
            save_processed_image(
                image = self._get_scaled_images(session, processed)[0],
                p = session.processing,
                processed = processed,
                output_dir = ensure_directory_exists(session.output.output_dir),
                file_name = None,
                archive_mode = self.archive_mode,
            )

        wait_until(lambda: not image_save_queue.busy)

    def _get_scaled_images(self, session: Session, processed: Processed) -> list[PILImage]:
        return [ensure_image_dims(x, size = (
            int(quantize(session.processing.width * self.scale, 8)),
            int(quantize(session.processing.height * self.scale, 8)),
        )) for x in processed.images] if self.scale != 1.0 else processed.images


class VideoRenderingModule(PipelineModule):
    id = "video_rendering"
    name = "Video rendering"

    render_draft_every_nth_frame: int = ui_param("Render draft every N-th frame", gr.Number, precision = 0, minimum = 1, step = 1, value = 100)
    render_final_every_nth_frame: int = ui_param("Render final every N-th frame", gr.Number, precision = 0, minimum = 1, step = 1, value = 1000)
    render_draft_on_finish: bool = ui_param("Render draft on finish", gr.Checkbox, value = False)
    render_final_on_finish: bool = ui_param("Render final on finish", gr.Checkbox, value = False)

    def forward(self, session: Session, processed: Processed, frame_index: int, frame_count: int, seed: int) -> Optional[Processed]:
        if frame_index % self.render_draft_every_nth_frame == 0:
            render_project_video(session.output.output_dir, session.output.project_subdir, session.video_renderer, False)

        if frame_index % self.render_final_every_nth_frame == 0:
            render_project_video(session.output.output_dir, session.output.project_subdir, session.video_renderer, True)

        return processed

    def finalize(self, session: Session, processed: Processed) -> None:
        if self.render_draft_on_finish:
            render_project_video(session.output.output_dir, session.output.project_subdir, session.video_renderer, False)

        if self.render_final_on_finish:
            render_project_video(session.output.output_dir, session.output.project_subdir, session.video_renderer, True)

        wait_until(lambda: not video_render_queue.busy)
