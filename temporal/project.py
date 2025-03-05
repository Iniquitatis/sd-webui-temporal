from pathlib import Path

from temporal.compat import get_latest_version, upgrade_project
from temporal.general_data import GeneralData
from temporal.iteration_data import IterationData
from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.pipeline import Pipeline


class Project(Serializable):
    version: int = Field(get_latest_version(), flags = {"private"})
    general: GeneralData = Field(factory = GeneralData)
    pipeline: Pipeline = Field(factory = Pipeline)
    iteration: IterationData = Field(factory = IterationData, flags = {"private"})

    @classmethod
    def load(cls, dir: Path) -> "Project":
        upgrade_project(dir)

        result = super().load(dir / "project")
        result.general.path = dir

        return result

    def save(self, dir: Path) -> None:
        super().save(dir / "project")

    def delete_session_data(self) -> None:
        for module in self.pipeline.modules:
            module.reset(self.general)

        self.iteration = IterationData()
