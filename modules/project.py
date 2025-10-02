from pathlib import Path
from typing_extensions import Self

from modules.compat import get_latest_version, upgrade_project
from modules.general_data import GeneralData
from modules.object import Field, Object
from modules.pipeline import Pipeline


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
