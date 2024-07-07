from collections import defaultdict
from os import environ
from pathlib import Path

from temporal.backend import Backend
from temporal.global_options import GlobalOptions
from temporal.preset_store import PresetStore
from temporal.project_store import ProjectStore
from temporal.video_renderer import VideoRenderer


class SharedData:
    def __init__(self) -> None:
        self.backend: Backend

        if environ["TEMPORAL_BACKEND"] == "WEBUI":
            from temporal.webui import WebUIBackend
            self.backend = WebUIBackend()
        else:
            raise NotImplementedError

    def init(self, options_path: Path, presets_path: Path) -> None:
        self.options = GlobalOptions()
        self.options.load(options_path)
        self.preset_store = PresetStore(presets_path, self.options.ui.preset_sorting_order)
        self.preset_store.refresh()
        self.project_store = ProjectStore(self.options.output.output_dir, self.options.ui.project_sorting_order)
        self.project_store.refresh()
        self.video_renderer = VideoRenderer()
        self.previewed_modules: defaultdict[str, bool] = defaultdict(lambda: True)


shared = SharedData()
