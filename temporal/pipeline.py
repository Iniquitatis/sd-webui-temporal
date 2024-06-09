from typing import Optional

from temporal.meta.serializable import Serializable, field
from temporal.pipeline_modules import PIPELINE_MODULES, PipelineModule
from temporal.session import Session
from temporal.utils.collection import reorder_dict
from temporal.utils.image import NumpyImage, np_to_pil
from temporal.web_ui import set_preview_image


class Pipeline(Serializable):
    module_order: list[str] = field(factory = list)
    modules: dict[str, PipelineModule] = field(factory = lambda: {id: cls() for id, cls in PIPELINE_MODULES.items()})

    def run(self, images: list[NumpyImage], session: Session, frame_index: int, frame_count: int, seed: int, show_only_finalized_frames: bool) -> Optional[list[NumpyImage]]:
        last_images = images

        for module in reorder_dict(self.modules, self.module_order).values():
            if not module.enabled:
                continue

            if not (last_images := module.forward(last_images, session, frame_index, frame_count, seed)):
                return None

            if not show_only_finalized_frames and module.preview:
                set_preview_image(np_to_pil(last_images[0]))

        if show_only_finalized_frames:
            set_preview_image(np_to_pil(last_images[0]))

        return last_images

    def finalize(self, images: list[NumpyImage], session: Session) -> None:
        for module in reorder_dict(self.modules, self.module_order).values():
            if not module.enabled:
                continue

            module.finalize(images, session)
