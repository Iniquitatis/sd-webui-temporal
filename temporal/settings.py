from pathlib import Path

from temporal.meta.configurable import Configurable, ConfigurableParam as Param
from temporal.meta.serializable import Serializable, SerializableField as Field


class OptionCategory(Configurable):
    pass


class OutputCategory(OptionCategory):
    name = "Output"

    output_dir: Path = Param("Output directory", value = Path("outputs/temporal"))
    autosave_every_n_iterations: int = Param("Autosave every N iterations", minimum = 1, step = 1, value = 10, ui_type = "box")


class UICategory(OptionCategory):
    name = "UI"

    preset_sorting_order: str = Param("Preset sorting order", choices = {"alphabetical": "Alphabetical", "date": "Date"}, value = "alphabetical", ui_type = "radio")
    project_sorting_order: str = Param("Project sorting order", choices = {"alphabetical": "Alphabetical", "date": "Date"}, value = "alphabetical", ui_type = "radio")
    gallery_size: int = Param("Gallery size", minimum = 1, maximum = 1000, step = 1, value = 10, ui_type = "box")


class Settings(Serializable):
    output: OutputCategory = Field(OutputCategory)
    ui: UICategory = Field(UICategory)
