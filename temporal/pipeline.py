import skimage

from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.pipeline_module import PipelineModule
from temporal.project import IterationData, Project
from temporal.shared import shared
from temporal.utils.collection import find_index_by_predicate
from temporal.utils.math import clamp


class Pipeline(Serializable):
    parallel: int = Field(1)
    modules: list[PipelineModule] = Field(factory = list)

    def run(self, project: Project) -> bool:
        if project.iteration.module_id is not None:
            skip_index = find_index_by_predicate(self.modules, lambda x: x.id == project.iteration.module_id)
        else:
            skip_index = -1

        for i, module in enumerate(self.modules):
            if i <= skip_index or not module.enabled:
                continue

            if not (images := module.forward(
                project.iteration.images,
                project,
                project.iteration.index,
                project.parameters.seed + project.iteration.index,
            )):
                return False

            project.iteration.images[:] = images
            project.iteration.module_id = module.id

            if shared.backend.is_interrupted():
                return False

            if not shared.options.live_preview.show_only_finished_images and shared.previewed_modules[module.id]:
                self._show_images(project.iteration)

        project.iteration.index += 1
        project.iteration.module_id = None

        if shared.options.live_preview.show_only_finished_images:
            self._show_images(project.iteration)

        return True

    def finalize(self, project: Project) -> None:
        for module in self.modules:
            if not module.enabled:
                continue

            module.finalize(project.iteration.images, project)

    def _show_images(self, iteration: IterationData) -> None:
        if shared.options.live_preview.preview_parallel_index == 0:
            preview = skimage.util.montage(iteration.images, channel_axis = -1)
        else:
            preview = iteration.images[clamp(shared.options.live_preview.preview_parallel_index - 1, 0, len(iteration.images) - 1)]

        shared.backend.set_preview(preview)
