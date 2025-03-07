from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

from temporal.backend import Backend
from temporal.fs_store import FSStore
from temporal.settings import Settings
from temporal.utils.image import NumpyImage, load_image, pil_to_np
from temporal.utils.modules import import_modules, list_modules_in_directory


class SharedData:
    def init(self, backend: Backend, settings_path: Path, presets_path: Path) -> None:
        # FIXME: SharedData -> Project -> Pipeline -> SharedData -> ...
        from temporal.preset import Preset
        from temporal.project import Project

        @dataclass
        class ExecutionState:
            active_project: Optional[Project] = None
            running: bool = False
            state: Literal["active", "stopping", "stopped"] = "stopped"
            current_iteration: int = 0
            total_iterations: int = 0
            preview: Optional[NumpyImage] = None

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
        self.state = ExecutionState()
        self.state_lock = Lock()

        import_modules(list_modules_in_directory("temporal/pipeline_modules", True, 4))
        import_modules(list_modules_in_directory("temporal/video_filters"))


shared = SharedData()
