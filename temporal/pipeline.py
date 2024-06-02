from typing import Optional

from modules.processing import Processed

from temporal.meta.serializable import Serializable, field
from temporal.pipeline_modules import PIPELINE_MODULES, PipelineModule
from temporal.session import Session
from temporal.utils.collection import reorder_dict
from temporal.web_ui import set_preview_image


class Pipeline(Serializable):
    module_order: list[str] = field(factory = list)
    modules: dict[str, PipelineModule] = field(factory = lambda: {id: cls() for id, cls in PIPELINE_MODULES.items()})

    def init(self, session: Session) -> None:
        for module in self.modules.values():
            module.init(session)

    def run(self, session: Session, processed: Processed, frame_index: int, frame_count: int, seed: int, show_only_finalized_frames: bool) -> Optional[Processed]:
        last_processed = processed

        for module in reorder_dict(self.modules, self.module_order).values():
            if not module.enabled:
                continue

            if not (last_processed := module.forward(session, last_processed, frame_index, frame_count, seed)):
                return None

            if not show_only_finalized_frames and module.preview:
                set_preview_image(last_processed.images[0])

        if show_only_finalized_frames:
            set_preview_image(last_processed.images[0])

        return last_processed

    def finalize(self, session: Session, processed: Processed) -> None:
        for module in reorder_dict(self.modules, self.module_order).values():
            if not module.enabled:
                continue

            module.finalize(session, processed)
