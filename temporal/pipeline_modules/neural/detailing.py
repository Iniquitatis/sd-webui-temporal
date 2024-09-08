from typing import Optional

from temporal.meta.configurable import EnumParam, FloatParam, IntParam
from temporal.pipeline_modules.neural import NeuralModule
from temporal.project import Project
from temporal.shared import shared
from temporal.utils.collection import get_first_element
from temporal.utils.image import NumpyImage, ensure_image_dims, np_to_pil, pil_to_np
from temporal.utils.math import quantize
from temporal.utils.object import copy_with_overrides


class DetailingModule(NeuralModule):
    name = "Detailing"

    scale: float = FloatParam("Scale", minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0, ui_type = "slider")
    sampler: str = EnumParam("Sampler", choices = shared.backend.list_samplers(), value = get_first_element(shared.backend.list_samplers()), ui_type = "menu")
    scheduler: str = EnumParam("Scheduler", choices = shared.backend.list_schedulers(), value = get_first_element(shared.backend.list_schedulers()), ui_type = "menu")
    steps: int = IntParam("Steps", minimum = 1, maximum = 150, step = 1, value = 15, ui_type = "slider")
    denoising_strength: float = FloatParam("Denoising strength", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.2, ui_type = "slider")

    def forward(self, images: list[NumpyImage], project: Project, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if not (processed_images := shared.backend.images_to_batches(
            copy_with_overrides(project.parameters,
                sampler = self.sampler,
                scheduler = self.scheduler,
                steps = self.steps,
                width = quantize(project.parameters.width * self.scale, 8),
                height = quantize(project.parameters.height * self.scale, 8),
                denoising_strength = self.denoising_strength,
            ),
            [(np_to_pil(x), seed + i, 1) for i, x in enumerate(images)],
            shared.options.processing.pixels_per_batch,
            shared.previewed_modules[self.id] and not shared.options.live_preview.show_only_finished_images,
        )):
            return None

        return [
            pil_to_np(ensure_image_dims(image_array[0], "RGB", (project.parameters.width, project.parameters.height)))
            for image_array in processed_images
        ]
