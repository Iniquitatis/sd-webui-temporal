import numpy as np

from temporal.meta.configurable import BoolParam
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.project import Project
from temporal.utils.image import NumpyImage


class SymmetryFilter(ImageFilter):
    id = "symmetry"
    name = "Symmetry"

    horizontal: bool = BoolParam("Horizontal", value = False)
    vertical: bool = BoolParam("Vertical", value = False)

    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        height, width = npim.shape[:2]
        npim = npim.copy()

        if self.horizontal:
            npim[:, width // 2:] = np.flip(npim[:, :width // 2], axis = 1)

        if self.vertical:
            npim[height // 2:, :] = np.flip(npim[:height // 2, :], axis = 0)

        return npim
