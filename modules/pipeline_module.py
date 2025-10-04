from typing import Any, Literal, Self

import skimage

from modules.animation import Animation
from modules.general_data import GeneralData
from modules.object import Field, Object, Static
from modules.pipeline_state import PipelineResult, PipelineState
from modules.seed import Seed
from modules.utils.image import NumpyImage, PILImage, make_trs_transform
from modules.utils.logging import log
from modules.video import Video


VisualizableType = NumpyImage | PILImage | Video


class PipelineModule(Object, abstract = True):
    name: str = Static("UNDEFINED")
    is_filter: bool = Static(False)
    is_visualizable: bool = Static(False)
    visualization_type: Literal["image", "video"] = Static("image")
    is_sampleable: bool = Static(False)
    sample_iterations: int = Static(1, flags = {"private"})

    enabled: bool = Field(True, name = "Enabled")
    preview: bool = Field(True, name = "Preview")
    animation: Animation = Field(Animation, name = "Animation")

    # NOTE: Hack to account for JSON not discerning between numeric types
    def validate(self, criteria: dict[str, Any]) -> Self:
        super().validate(criteria)

        for track in self.animation.tracks:
            field_type = self.__fields__[track.key].type

            for keyframe in track.keyframes:
                if field_type == int and isinstance(keyframe.value, float):
                    log.debug(f"Fixing {track.key}/{keyframe.frame}: got {keyframe.value.__class__.__name__}, expected {field_type.__name__}")
                    keyframe.value = int(keyframe.value)

                elif field_type == float and isinstance(keyframe.value, int):
                    log.debug(f"Fixing {track.key}/{keyframe.frame}: got {keyframe.value.__class__.__name__}, expected {field_type.__name__}")
                    keyframe.value = float(keyframe.value)

        return self

    def forward(self, image: NumpyImage, general: GeneralData) -> PipelineResult:
        yield PipelineState.finish(image = image, preview = self.preview)

    def finalize(self, general: GeneralData) -> None:
        pass

    def interrupt(self, general: GeneralData) -> None:
        pass

    def visualize(self, general: GeneralData) -> VisualizableType:
        raise NotImplementedError

    def execute(self, image: NumpyImage) -> NumpyImage:
        for state in self.forward(image, GeneralData(initial_image = image, seed = Seed(31337))):
            match state:
                case PipelineState.progress():
                    pass
                case PipelineState.finish():
                    return state.image
                case PipelineState.fail():
                    log.warning("Couldn't execute a module:", state.message)
                    break

        return image

    def sample(self, image: NumpyImage) -> NumpyImage:
        last_image = image.copy()

        for i in range(self.sample_iterations):
            last_image = self.execute(last_image)

            if (i + 1) != self.sample_iterations:
                last_image = skimage.transform.warp(last_image, make_trs_transform(
                    image_size = (image.shape[1], image.shape[0]),
                    translation = (0.01, 0.01),
                    rotation = 3.0,
                    scale = 0.99,
                ), mode = "symmetric")

        return last_image
