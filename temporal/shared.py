from collections import defaultdict
from pathlib import Path

from temporal.global_options import GlobalOptions
from temporal.preset_store import PresetStore
from temporal.project_store import ProjectStore
from temporal.video_renderer import VideoRenderer


class SharedData:
    def init(self, options_path: Path, presets_path: Path) -> None:
        self.options = GlobalOptions()
        self.options.load(options_path)
        self.preset_store = PresetStore(presets_path)
        self.preset_store.refresh_presets()
        self.project_store = ProjectStore(Path(self.options.output.output_dir))
        self.project_store.refresh_projects()
        self.video_renderer = VideoRenderer()
        self.previewed_modules: defaultdict[str, bool] = defaultdict(lambda: True)


shared = SharedData()
