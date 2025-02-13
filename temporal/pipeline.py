import skimage

from temporal.iteration_data import IterationData
from temporal.project import Project
from temporal.shared import shared
from temporal.utils.math import clamp


class Pipeline:
    def run(self, project: Project) -> bool:
        for i, module in enumerate(project.modules):
            if i < project.iteration.step or not module.enabled:
                continue

            if not (images := module.forward(
                project.iteration.images,
                project.general,
                project.iteration.index,
                project.general.parameters.seed + project.iteration.index,
            )):
                return False

            project.iteration.images[:] = images
            project.iteration.step += 1

            if shared.backend.is_interrupted():
                return False

            if not shared.options.live_preview.show_only_finished_images and shared.previewed_modules[module.id]:
                self._show_images(project.iteration)

        project.iteration.index += 1
        project.iteration.step = 0

        if shared.options.live_preview.show_only_finished_images:
            self._show_images(project.iteration)

        return True

    def finalize(self, project: Project) -> None:
        for module in project.modules:
            if not module.enabled:
                continue

            module.finalize(project.iteration.images, project.general)

    def _show_images(self, iteration: IterationData) -> None:
        if shared.options.live_preview.preview_parallel_index == 0:
            preview = skimage.util.montage(iteration.images, channel_axis = -1)
        else:
            preview = iteration.images[clamp(shared.options.live_preview.preview_parallel_index - 1, 0, len(iteration.images) - 1)]

        shared.backend.set_preview(preview)
