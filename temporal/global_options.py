from pathlib import Path

import gradio as gr

from temporal.meta.configurable import Configurable, ui_param
from temporal.meta.serializable import Serializable, field


class OptionCategory(Configurable):
    name: str = "UNDEFINED"


class OutputCategory(OptionCategory):
    name = "Output"

    output_dir: Path = ui_param("Output directory", gr.Textbox, preprocessor = Path, postprocessor = Path.as_posix, value = Path("outputs/temporal"))
    autosave_every_n_iterations: int = ui_param("Autosave every N iterations", gr.Number, precision = 0, minimum = 1, step = 1, value = 10)


class LivePreviewCategory(OptionCategory):
    name = "Live preview"

    show_only_finished_images: bool = ui_param("Show only finished images", gr.Checkbox, value = False)
    preview_parallel_index: int = ui_param("Parallel index for preview", gr.Number, precision = 0, minimum = 0, step = 1, value = 1)


class ProcessingCategory(OptionCategory):
    name = "Processing"

    pixels_per_batch: int = ui_param("Pixels per batch", gr.Number, precision = 0, minimum = 4096, step = 4096, value = 1048576)


class UICategory(OptionCategory):
    name = "UI"

    preset_sorting_order: str = ui_param("Preset sorting order", gr.Dropdown, choices = ["alphabetical", "date"], value = "alphabetical")
    project_sorting_order: str = ui_param("Project sorting order", gr.Dropdown, choices = ["alphabetical", "date"], value = "alphabetical")
    gallery_size: int = ui_param("Gallery size", gr.Number, precision = 0, minimum = 1, maximum = 1000, step = 1, value = 10)


class GlobalOptions(Serializable):
    output: OutputCategory = field(factory = OutputCategory)
    live_preview: LivePreviewCategory = field(factory = LivePreviewCategory)
    processing: ProcessingCategory = field(factory = ProcessingCategory)
    ui: UICategory = field(factory = UICategory)
