from typing import Iterator, Optional, TypeVar

from temporal.animation import Animation
from temporal.general_data import GeneralData
from temporal.object import Field, Object
from temporal.pipeline_module import PipelineModule
from temporal.utils.image import NumpyImage
from temporal.utils.object import set_property_by_path


T = TypeVar("T", bound = PipelineModule)


class Pipeline(Object):
    modules: list[PipelineModule] = Field(list, name = "Modules")
    animation: Animation = Field(Animation, name = "Animation")

    def run(self, image: NumpyImage, general: GeneralData, iter_index: int) -> Iterator[tuple[int, Optional[NumpyImage], bool]]:
        for path, value in self.animation.evaluate(iter_index).items():
            set_property_by_path(self, path, value)

        last_image = image

        for i, module in enumerate(self.modules):
            if not module.enabled:
                continue

            if (last_image := module.forward(
                last_image,
                general,
                iter_index,
                general.seed.fixed_value + iter_index * len(self.modules) + i,
            )) is not None:
                yield i, last_image, module.preview
            else:
                yield i, None, False
                return

    def finalize(self, image: NumpyImage, general: GeneralData) -> None:
        for module in self.modules:
            if not module.enabled:
                continue

            module.finalize(image, general)

    def reset(self, general: GeneralData) -> None:
        for module in self.modules:
            module.reset(general)
