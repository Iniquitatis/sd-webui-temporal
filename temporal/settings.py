from pathlib import Path

from temporal.object import Field, Object, Static


class OptionCategory(Object):
    name: str = Static("UNDEFINED")


class FileSystemCategory(OptionCategory):
    name = "File system"

    preset_dir: Path = Field(Path("presets"), name = "Preset directory")
    preset_sorting_order: str = Field("alphabetical", name = "Preset sorting order", choices = {"alphabetical": "Alphabetical", "date": "Date"}, display = "radio")
    project_dir: Path = Field(Path("outputs/temporal"), name = "Project directory")
    project_sorting_order: str = Field("alphabetical", name = "Project sorting order", choices = {"alphabetical": "Alphabetical", "date": "Date"}, display = "radio")


class ExecutionCategory(OptionCategory):
    name = "Execution"

    autosave_every_nth_iteration: int = Field(10, name = "Autosave every N-th iteration", minimum = 1, step = 1, display = "box")


class Settings(Object):
    fs: FileSystemCategory = Field(FileSystemCategory, name = "File system", display = "accordion")
    execution: ExecutionCategory = Field(ExecutionCategory, name = "Execution", display = "accordion")
