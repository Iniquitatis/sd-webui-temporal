from abc import abstractmethod

from modules.general_data import GeneralData
from modules.pipeline_module import PipelineModule
from modules.pipeline_state import PipelineResult, PipelineState
from modules.utils.image import NumpyImage


class ToolModule(PipelineModule, abstract = True):
    def forward(self, image: NumpyImage, general: GeneralData) -> PipelineResult:
        self.process(image, general)
        yield PipelineState.finish(image = image, preview = self.preview)

    @abstractmethod
    def process(self, image: NumpyImage, general: GeneralData) -> None:
        raise NotImplementedError
