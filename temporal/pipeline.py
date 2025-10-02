from temporal.general_data import GeneralData
from temporal.object import Field, Object
from temporal.pipeline_module import PipelineModule
from temporal.pipeline_state import PipelineResult, PipelineState
from temporal.utils.image import NumpyImage


class Pipeline(Object):
    modules: list[PipelineModule] = Field(list, name = "Modules")
    iteration: int = Field(0, flags = {"runtime"})

    def run(self, image: NumpyImage, general: GeneralData) -> PipelineResult:
        result = PipelineState.finish(image = image, preview = True)

        for module in self.modules:
            # NOTE: Intentionally put before the enabled check, as it can change
            # the enabled state
            for key, value in module.animation.evaluate(self.iteration + 1).items():
                setattr(module, key, value)

            if not module.enabled:
                continue

            for state in module.forward(result.image, general):
                match state:
                    case PipelineState.progress():
                        yield state
                    case PipelineState.finish():
                        result = state
                        yield PipelineState.progress(image = state.image, preview = state.preview)
                    case PipelineState.fail():
                        yield state
                        return

        self.iteration += 1

        yield result

    def finalize(self, general: GeneralData) -> None:
        for module in self.modules:
            if module.enabled:
                module.finalize(general)

    def interrupt(self, general: GeneralData) -> None:
        for module in self.modules:
            module.interrupt(general)
