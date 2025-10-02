from pathlib import Path
from typing_extensions import Self

from temporal.compat import get_latest_version, upgrade_project
from temporal.general_data import GeneralData
from temporal.object import Field, Object
from temporal.pipeline import Pipeline


class Project(Object):
    general: GeneralData = Field(GeneralData, name = "General", display = "tab")
    pipeline: Pipeline = Field(Pipeline, name = "Pipeline", display = "tab")
    version: int = Field(get_latest_version(), flags = {"private"})

    @classmethod
    def load(cls, dir: Path) -> Self:
        upgrade_project(dir)
        return super().load(dir / "project")

    def save(self, dir: Path) -> None:
        super().save(dir / "project")
