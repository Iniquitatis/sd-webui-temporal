import skimage

from modules import shared as webui_shared
from modules.shared_state import State

from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.pipeline_modules import PIPELINE_MODULES, PipelineModule
from temporal.session import IterationData, Session
from temporal.shared import shared
from temporal.utils.collection import reorder_dict
from temporal.utils.image import np_to_pil


# FIXME: To shut up the type checker
state: State = getattr(webui_shared, "state")


class Pipeline(Serializable):
    parallel: int = Field(1)
    module_order: list[str] = Field(factory = list)
    modules: dict[str, PipelineModule] = Field(factory = lambda: {id: cls() for id, cls in PIPELINE_MODULES.items()})

    def run(self, session: Session) -> bool:
        ordered_modules = reorder_dict(self.modules, self.module_order)
        ordered_keys = list(ordered_modules.keys())

        if session.iteration.module_id is not None:
            skip_index = ordered_keys.index(session.iteration.module_id)
        else:
            skip_index = -1

        for i, module in enumerate(ordered_modules.values()):
            if i <= skip_index or not module.enabled:
                continue

            if not (images := module.forward(
                session.iteration.images,
                session,
                session.iteration.index,
                session.processing.seed + session.iteration.index,
            )):
                return False

            session.iteration.images[:] = images
            session.iteration.module_id = module.id

            if state.interrupted or state.skipped:
                return False

            if not shared.options.live_preview.show_only_finished_images and shared.previewed_modules[module.id]:
                self._show_images(session.iteration)

        session.iteration.index += 1
        session.iteration.module_id = None

        if shared.options.live_preview.show_only_finished_images:
            self._show_images(session.iteration)

        return True

    def finalize(self, session: Session) -> None:
        for module in reorder_dict(self.modules, self.module_order).values():
            if not module.enabled:
                continue

            module.finalize(session.iteration.images, session)

    def _show_images(self, iteration: IterationData) -> None:
        if shared.options.live_preview.preview_parallel_index == 0:
            preview = skimage.util.montage(iteration.images, channel_axis = -1)
        else:
            preview = iteration.images[min(max(shared.options.live_preview.preview_parallel_index - 1, 0), len(iteration.images) - 1)]

        state.assign_current_image(np_to_pil(preview))
