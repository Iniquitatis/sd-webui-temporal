from typing import Optional

from temporal.meta.configurable import EnumParam, FloatParam, IntParam
from temporal.pipeline_modules.neural import NeuralModule
from temporal.project import Project
from temporal.shared import shared
from temporal.utils.collection import get_first_element
from temporal.utils.image import NumpyImage, ensure_image_dims, np_to_pil, pil_to_np
from temporal.utils.math import quantize


class DetailingModule(NeuralModule):
    name = "Detailing"

    scale: float = FloatParam("Scale", minimum = 0.25, maximum = 4.0, step = 0.25, value = 1.0, ui_type = "slider")
    sampler: str = EnumParam("Sampling method", choices = shared.backend.samplers, value = get_first_element(shared.backend.samplers, ""), ui_type = "menu")
    if shared.backend.has_schedulers:
        scheduler: str = EnumParam("Schedule type", choices = shared.backend.schedulers, value = get_first_element(shared.backend.schedulers, ""), ui_type = "menu")
    steps: int = IntParam("Steps", minimum = 1, maximum = 150, step = 1, value = 15, ui_type = "slider")
    denoising_strength: float = FloatParam("Denoising strength", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.2, ui_type = "slider")

    def forward(self, images: list[NumpyImage], project: Project, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if not (processed_images := shared.backend.process_batches(
            [(np_to_pil(x), seed + i, 1) for i, x in enumerate(images)],
            shared.options.processing.pixels_per_batch,
            shared.previewed_modules[self.id] and not shared.options.live_preview.show_only_finished_images,
            sampler = self.sampler,
            steps = self.steps,
            width = quantize(project.backend_data.width * self.scale, 8),
            height = quantize(project.backend_data.height * self.scale, 8),
            denoising_strength = self.denoising_strength,
            **({"scheduler" : self.scheduler} if hasattr(self, "scheduler") else {})
        )):
            return None

        return [
            pil_to_np(ensure_image_dims(image_array[0], "RGB", (project.backend_data.width, project.backend_data.height)))
            for image_array in processed_images
        ]
