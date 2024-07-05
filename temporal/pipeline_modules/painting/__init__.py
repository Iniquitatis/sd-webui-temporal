from abc import abstractmethod

from temporal.pipeline_modules.filtering import ImageFilter
from temporal.project import Project
from temporal.utils.image import NumpyImage, alpha_blend


class PaintingModule(ImageFilter, abstract = True):
    icon = "\U0001f58c\ufe0f"

    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        return alpha_blend(npim, self.draw((npim.shape[1], npim.shape[0]), parallel_index, project, frame_index, seed))

    @abstractmethod
    def draw(self, size: tuple[int, int], parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        raise NotImplementedError
