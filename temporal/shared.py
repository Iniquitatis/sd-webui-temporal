from collections import defaultdict
from pathlib import Path

from temporal.backend import Backend
from temporal.global_options import GlobalOptions
from temporal.video_renderer import VideoRenderer


class SharedData:
    def init(self, backend: Backend, options_path: Path, presets_path: Path) -> None:
        # FIXME: SharedData -> PresetStore -> Preset -> Project -> Pipeline -> SharedData -> PresetStore -> ...
        from temporal.preset_store import PresetStore
        from temporal.project_store import ProjectStore

        self.backend = backend
        self.options_path = options_path
        self.presets_path = presets_path
        self.options = GlobalOptions.load(options_path)
        self.preset_store = PresetStore(presets_path, self.options.ui.preset_sorting_order)
        self.preset_store.refresh()
        self.project_store = ProjectStore(self.options.output.output_dir, self.options.ui.project_sorting_order)
        self.project_store.refresh()
        self.video_renderer = VideoRenderer()
        self.previewed_modules: defaultdict[str, bool] = defaultdict(lambda: True)


shared = SharedData()
