from typing import Optional

import numpy as np

from temporal.meta.configurable import FloatParam, IntParam
from temporal.pipeline_modules.neural import NeuralModule
from temporal.project import Project
from temporal.shared import shared
from temporal.utils.image import NumpyImage, np_to_pil, pil_to_np
from temporal.utils.numpy import average_array, make_eased_weight_array, saturate_array
from temporal.utils.object import copy_with_overrides
from temporal.web_ui import process_images


class ProcessingModule(NeuralModule):
    id = "processing"
    name = "Processing"

    samples: int = IntParam("Sample count", minimum = 1, value = 1, ui_type = "box")
    trimming: float = FloatParam("Trimming", minimum = 0.0, maximum = 0.5, step = 0.01, value = 0.0, ui_type = "slider")
    easing: float = FloatParam("Easing", minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0, ui_type = "slider")
    preference: float = FloatParam("Preference", minimum = -2.0, maximum = 2.0, step = 0.1, value = 0.0, ui_type = "slider")

    def forward(self, images: list[NumpyImage], project: Project, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        if not (processed_images := process_images(
            copy_with_overrides(project.processing, do_not_save_samples = True, do_not_save_grid = True),
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
