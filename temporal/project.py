from pathlib import Path

from temporal.animation import Animation
from temporal.compat import get_latest_version, upgrade_project
from temporal.general_data import GeneralData
from temporal.initial_noise_params import InitialNoiseParams
from temporal.iteration_data import IterationData
from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.pipeline_module import PipelineModule


class Project(Serializable):
    version: int = Field(get_latest_version())
    general: GeneralData = Field(factory = GeneralData)
    initial_noise: InitialNoiseParams = Field(factory = InitialNoiseParams)
    modules: list[PipelineModule] = Field(factory = list)
    animation: Animation = Field(factory = Animation)
    iteration: IterationData = Field(factory = IterationData)

    def load(self, dir: Path) -> None:
        upgrade_project(dir)
        super().load(dir / "project")

    def save(self, dir: Path) -> None:
        super().save(dir / "project")

    def delete_session_data(self) -> None:
        for module in self.modules:
            module.reset()

        self.iteration = IterationData()
