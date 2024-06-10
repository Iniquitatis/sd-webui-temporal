from modules import shared
from modules.shared_state import State

from temporal.meta.serializable import Serializable, field
from temporal.pipeline_modules import PIPELINE_MODULES, PipelineModule
from temporal.session import Session
from temporal.utils.collection import reorder_dict
from temporal.utils.image import np_to_pil
from temporal.web_ui import set_preview_image


# FIXME: To shut up the type checker
state: State = getattr(shared, "state")


class Pipeline(Serializable):
    parallel: int = field(1)
    module_order: list[str] = field(factory = list)
    modules: dict[str, PipelineModule] = field(factory = lambda: {id: cls() for id, cls in PIPELINE_MODULES.items()})

    def run(self, session: Session, show_only_finalized_frames: bool) -> bool:
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

            if not show_only_finalized_frames and module.preview:
                set_preview_image(np_to_pil(session.iteration.images[0]))

        session.iteration.index += 1
        session.iteration.module_id = None

        if show_only_finalized_frames:
            set_preview_image(np_to_pil(session.iteration.images[0]))

        return True

    def finalize(self, session: Session) -> None:
        for module in reorder_dict(self.modules, self.module_order).values():
            if not module.enabled:
                continue

            module.finalize(session.iteration.images, session)
