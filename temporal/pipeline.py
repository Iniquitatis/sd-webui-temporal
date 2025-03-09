from typing import Optional, Type, TypeVar

from temporal.animation import Animation
from temporal.general_data import GeneralData
from temporal.iteration_data import IterationData
from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.pipeline_module import PipelineModule
from temporal.shared import shared
from temporal.utils.collection import find_by_predicate
from temporal.utils.object import set_property_by_path


T = TypeVar("T", bound = PipelineModule)


class Pipeline(Serializable):
    modules: list[PipelineModule] = Field(list)
    animation: Animation = Field(Animation)

    def find_module(self, uuid: str, type: Type[T] = PipelineModule) -> Optional[T]:
        if isinstance(result := find_by_predicate(self.modules, lambda x: x.uuid == uuid), type):
            return result

    def run(self, general: GeneralData, iteration: IterationData) -> bool:
        for path, value in self.animation.evaluate(iteration.index).items():
            set_property_by_path(self, path, value)

        for i, module in enumerate(self.modules):
            if i < iteration.step or not module.enabled:
                continue

            # FIXME: Optional thing
            if (image := module.forward(
                iteration.image,
                general,
                iteration.index,
                general.seed + iteration.index * len(self.modules) + iteration.step,
            )) is None:
                return False

            iteration.image = image
            iteration.step += 1

            with shared.state_lock:
                if not shared.state.running:
                    return False

                if module.preview:
                    shared.state.preview = iteration.image

        iteration.index += 1
        iteration.step = 0

        return True

    def finalize(self, general: GeneralData, iteration: IterationData) -> None:
        for module in self.modules:
            if not module.enabled:
                continue

            # FIXME: Optional thing
            module.finalize(iteration.image, general)
