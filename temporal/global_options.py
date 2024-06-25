from pathlib import Path

from temporal.meta.configurable import BoolParam, Configurable, EnumParam, IntParam, PathParam
from temporal.meta.serializable import Serializable, SerializableField as Field


class OptionCategory(Configurable):
    name: str = "UNDEFINED"


class OutputCategory(OptionCategory):
    name = "Output"

    output_dir: Path = PathParam("Output directory", value = Path("outputs/temporal"))
    autosave_every_n_iterations: int = IntParam("Autosave every N iterations", minimum = 1, step = 1, value = 10, ui_type = "box")


class LivePreviewCategory(OptionCategory):
    name = "Live preview"

    show_only_finished_images: bool = BoolParam("Show only finished images", value = False)
    preview_parallel_index: int = IntParam("Parallel index for preview", minimum = 0, step = 1, value = 1, ui_type = "box")


class ProcessingCategory(OptionCategory):
    name = "Processing"

    pixels_per_batch: int = IntParam("Pixels per batch", minimum = 4096, step = 4096, value = 1048576, ui_type = "box")


class UICategory(OptionCategory):
    name = "UI"

    preset_sorting_order: str = EnumParam("Preset sorting order", choices = [("alphabetical", "Alphabetical"), ("date", "Date")], value = "alphabetical", ui_type = "radio")
    project_sorting_order: str = EnumParam("Project sorting order", choices = [("alphabetical", "Alphabetical"), ("date", "Date")], value = "alphabetical", ui_type = "radio")
    gallery_size: int = IntParam("Gallery size", minimum = 1, maximum = 1000, step = 1, value = 10, ui_type = "box")


class GlobalOptions(Serializable):
    output: OutputCategory = Field(factory = OutputCategory)
    live_preview: LivePreviewCategory = Field(factory = LivePreviewCategory)
    processing: ProcessingCategory = Field(factory = ProcessingCategory)
    ui: UICategory = Field(factory = UICategory)
