from abc import abstractmethod

from temporal.general_data import GeneralData
from temporal.pipeline_module import PipelineModule
from temporal.pipeline_state import PipelineResult, PipelineState
from temporal.utils.image import NumpyImage


class TemporalModule(PipelineModule, abstract = True):
    is_sampleable = True
    sample_iterations = 10

    def forward(self, image: NumpyImage, general: GeneralData) -> PipelineResult:
        yield PipelineState.finish(image = self.process(image, general), preview = self.preview)

    @abstractmethod
    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        raise NotImplementedError
