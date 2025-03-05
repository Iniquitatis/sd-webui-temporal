from abc import abstractmethod
from typing import Optional

from temporal.general_data import GeneralData
from temporal.pipeline_module import PipelineModule
from temporal.utils.image import NumpyImage


class ToolModule(PipelineModule, abstract = True):
    def forward(self, image: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> Optional[NumpyImage]:
        self.process(image, general, frame_index, seed)
        return image

    @abstractmethod
    def process(self, image: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> None:
        raise NotImplementedError
