from pathlib import Path

from temporal.object import Field, Meta, Object, Param


class OptionCategory(Object):
    name: str = Meta("UNDEFINED")


class FileSystemCategory(OptionCategory):
    name = "File system"

    preset_dir: Path = Param("Preset directory", value = Path("presets"))
    preset_sorting_order: str = Param("Preset sorting order", choices = {"alphabetical": "Alphabetical", "date": "Date"}, value = "alphabetical", ui_type = "radio")
    project_dir: Path = Param("Project directory", value = Path("outputs/temporal"))
    project_sorting_order: str = Param("Project sorting order", choices = {"alphabetical": "Alphabetical", "date": "Date"}, value = "alphabetical", ui_type = "radio")


class ExecutionCategory(OptionCategory):
    name = "Execution"

    autosave_every_nth_iteration: int = Param("Autosave every N-th iteration", minimum = 1, step = 1, value = 10, ui_type = "box")


class Settings(Object):
    fs: FileSystemCategory = Field(FileSystemCategory)
    execution: ExecutionCategory = Field(ExecutionCategory)
