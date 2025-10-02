from pathlib import Path

from modules.object import Field, Object, Static


class OptionCategory(Object):
    name: str = Static("UNDEFINED")


class FileSystemCategory(OptionCategory):
    name = "File system"

    project_dir: Path = Field(Path("outputs"), name = "Project directory")
    project_sorting_order: str = Field("alphabetical", name = "Project sorting order", choices = {"alphabetical": "Alphabetical", "date": "Date"}, display = "radio")


class Settings(Object):
    fs: FileSystemCategory = Field(FileSystemCategory, name = "File system", display = "accordion")
