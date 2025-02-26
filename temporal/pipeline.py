import skimage

from temporal.animation import Animation
from temporal.general_data import GeneralData
from temporal.iteration_data import IterationData
from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.pipeline_module import PipelineModule
from temporal.shared import shared
from temporal.utils.math import clamp
from temporal.utils.object import set_property_by_path


class Pipeline(Serializable):
    modules: list[PipelineModule] = Field(factory = list)
    animation: Animation = Field(factory = Animation)

    def run(self, general: GeneralData, iteration: IterationData) -> bool:
        for path, value in self.animation.evaluate(iteration.index).items():
            set_property_by_path(self, path, value)

        for i, module in enumerate(self.modules):
            if i < iteration.step or not module.enabled:
                continue

            if not (images := module.forward(
                iteration.images,
                general,
                iteration.index,
                general.seed + iteration.index * len(self.modules) + iteration.step,
            )):
                return False

            iteration.images[:] = images
            iteration.step += 1

            # FIXME: Gives issues when any non-Processing module is first in the
            # module list
            # if shared.backend.is_interrupted():
            #     return False

            if not shared.options.live_preview.show_only_finished_images and shared.previewed_modules[module.id]:
                self._show_images(iteration)

        iteration.index += 1
        iteration.step = 0

        if shared.options.live_preview.show_only_finished_images:
            self._show_images(iteration)

        return True

    def finalize(self, general: GeneralData, iteration: IterationData) -> None:
        for module in self.modules:
            if not module.enabled:
                continue

            module.finalize(iteration.images, general)

    def _show_images(self, iteration: IterationData) -> None:
        if shared.options.live_preview.preview_parallel_index == 0:
            preview = skimage.util.montage(iteration.images, channel_axis = -1)
        else:
            preview = iteration.images[clamp(shared.options.live_preview.preview_parallel_index - 1, 0, len(iteration.images) - 1)]

        shared.backend.set_preview(preview)
