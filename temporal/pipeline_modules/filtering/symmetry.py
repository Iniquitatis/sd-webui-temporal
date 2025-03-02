import numpy as np

from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage


class SymmetryFilter(ImageFilter):
    name = "Symmetry"

    horizontal: bool = Param("Horizontal", value = False)
    vertical: bool = Param("Vertical", value = False)

    def process(self, npim: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        height, width = npim.shape[:2]
        npim = npim.copy()

        if self.horizontal:
            npim[:, width // 2:] = np.flip(npim[:, :width // 2], axis = 1)

        if self.vertical:
            npim[height // 2:, :] = np.flip(npim[:height // 2, :], axis = 0)

        return npim
