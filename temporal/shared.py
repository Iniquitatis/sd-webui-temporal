from collections import defaultdict
from pathlib import Path

from temporal.backend import Backend
from temporal.fs_store import FSStore
from temporal.settings import Settings
from temporal.utils.image import load_image, pil_to_np
from temporal.utils.modules import import_modules, list_modules_in_directory


class SharedData:
    def init(self, backend: Backend, settings_path: Path, presets_path: Path) -> None:
        # FIXME: SharedData -> Project -> Pipeline -> SharedData -> ...
        from temporal.preset import Preset
        from temporal.project import Project

        self.backend = backend
        self.settings_path = settings_path
        self.presets_path = presets_path
        self.settings = Settings.load(settings_path)
        self.preset_store = FSStore(Preset, presets_path, self.settings.ui.preset_sorting_order)
        self.preset_store.refresh()
        self.project_store = FSStore(Project, self.settings.output.output_dir, self.settings.ui.project_sorting_order)
        self.project_store.refresh()
        self.previewed_modules: defaultdict[str, bool] = defaultdict(lambda: True)
        self.sample_image = pil_to_np(load_image("data/sample_image.png"))

        import_modules(list_modules_in_directory("temporal/pipeline_modules", True, 4))


shared = SharedData()
