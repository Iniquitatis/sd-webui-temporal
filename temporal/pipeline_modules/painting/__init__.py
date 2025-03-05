from abc import abstractmethod

from temporal.general_data import GeneralData
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage, alpha_blend


class PaintingModule(ImageFilter, abstract = True):
    def process(self, image: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        return alpha_blend(image, self.draw((image.shape[1], image.shape[0]), general, frame_index, seed))

    @abstractmethod
    def draw(self, size: tuple[int, int], general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        raise NotImplementedError
