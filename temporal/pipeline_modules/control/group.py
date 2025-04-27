from typing import Optional

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_module import PipelineModule
from temporal.pipeline_modules.control import ControlModule
from temporal.utils.image import NumpyImage


class GroupModule(ControlModule):
    name = "Group"

    modules: list[PipelineModule] = Field(list, name = "Modules")

    def forward(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> Optional[NumpyImage]:
        last_image = image

        for module in self.modules:
            if not module.enabled:
                continue

            if (last_image := module.forward(last_image, general, iter_index, seed)) is None:
                return None

        return last_image

    def finalize(self, image: NumpyImage, general: GeneralData) -> None:
        for module in self.modules:
            if not module.enabled:
                continue

            module.finalize(image, general)
