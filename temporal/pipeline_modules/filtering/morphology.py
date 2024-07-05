import skimage

from temporal.meta.configurable import EnumParam, IntParam
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.project import Project
from temporal.utils.image import NumpyImage, apply_channelwise


class MorphologyFilter(ImageFilter):
    id = "morphology"
    name = "Morphology"

    mode: str = EnumParam("Mode", choices = [
        ("erosion", "Erosion"),
        ("dilation", "Dilation"),
        ("opening", "Opening"),
        ("closing", "Closing"),
    ], value = "erosion", ui_type = "menu")
    radius: int = IntParam("Radius", minimum = 0, maximum = 50, step = 1, value = 0, ui_type = "slider")

    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        func = (
            skimage.morphology.erosion  if self.mode == "erosion"  else
            skimage.morphology.dilation if self.mode == "dilation" else
            skimage.morphology.opening  if self.mode == "opening"  else
            skimage.morphology.closing  if self.mode == "closing"  else
            lambda image, footprint: image
        )
        footprint = skimage.morphology.disk(self.radius)
        return apply_channelwise(npim, lambda x: func(x, footprint))
