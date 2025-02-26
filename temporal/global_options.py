from pathlib import Path

from temporal.meta.configurable import Configurable, ConfigurableParam as Param
from temporal.meta.serializable import Serializable, SerializableField as Field


class OptionCategory(Configurable):
    pass


class OutputCategory(OptionCategory):
    name = "Output"

    output_dir: Path = Param("Output directory", value = Path("outputs/temporal"))
    autosave_every_n_iterations: int = Param("Autosave every N iterations", minimum = 1, step = 1, value = 10, ui_type = "box")


class LivePreviewCategory(OptionCategory):
    name = "Live preview"

    show_only_finished_images: bool = Param("Show only finished images", value = False)
    preview_parallel_index: int = Param("Parallel index for preview", minimum = 0, step = 1, value = 1, ui_type = "box")


class ProcessingCategory(OptionCategory):
    name = "Processing"

    pixels_per_batch: int = Param("Pixels per batch", minimum = 4096, step = 4096, value = 1048576, ui_type = "box")


class UICategory(OptionCategory):
    name = "UI"

    preset_sorting_order: str = Param("Preset sorting order", choices = [("alphabetical", "Alphabetical"), ("date", "Date")], value = "alphabetical", ui_type = "radio")
    project_sorting_order: str = Param("Project sorting order", choices = [("alphabetical", "Alphabetical"), ("date", "Date")], value = "alphabetical", ui_type = "radio")
    gallery_size: int = Param("Gallery size", minimum = 1, maximum = 1000, step = 1, value = 10, ui_type = "box")


# TODO: Rename to Settings
class GlobalOptions(Serializable):
    output: OutputCategory = Field(factory = OutputCategory)
    live_preview: LivePreviewCategory = Field(factory = LivePreviewCategory)
    processing: ProcessingCategory = Field(factory = ProcessingCategory)
    ui: UICategory = Field(factory = UICategory)
