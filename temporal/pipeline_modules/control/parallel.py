from typing import Optional

import numpy as np

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_module import PipelineModule
from temporal.pipeline_modules.control import ControlModule
from temporal.utils.image import NumpyImage
from temporal.utils.numpy import average_array, make_eased_weight_array, saturate_array


class ParallelModule(ControlModule):
    name = "Parallel"

    count: int = Field(1, name = "Count", minimum = 1, step = 1, display = "box")
    trimming: float = Field(0.0, name = "Trimming", minimum = 0.0, maximum = 0.5, step = 0.01, display = "slider")
    easing: float = Field(0.0, name = "Easing", minimum = 0.0, maximum = 16.0, step = 0.1, display = "slider")
    preference: float = Field(0.0, name = "Preference", minimum = -2.0, maximum = 2.0, step = 0.1, display = "slider")
    modules: list[PipelineModule] = Field(list, name = "Modules")

    def forward(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> Optional[NumpyImage]:
        images = []

        for _ in range(self.count):
            last_image = None

            for module in self.modules:
                if not module.enabled:
                    continue

                if (last_image := module.forward(image, general, iter_index, seed)) is None:
                    return None

            if last_image is None:
                return None

            images.append(last_image)

        return images[0] if self.count == 1 else saturate_array(average_array(
            np.array(images),
            axis = 0,
            trim = self.trimming,
            power = self.preference + 1.0,
            weights = make_eased_weight_array(self.count, self.easing),
        ))

    def finalize(self, image: NumpyImage, general: GeneralData) -> None:
        for module in self.modules:
            if not module.enabled:
                continue

            module.finalize(image, general)
