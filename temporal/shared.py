from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

from temporal.backend import Backend
from temporal.fs_store import FSStore
from temporal.settings import Settings
from temporal.utils.image import NumpyImage, load_image, pil_to_np
from temporal.utils.modules import import_modules, list_modules_in_directory


class SharedData:
    def init(self, backend: Backend, settings_path: Path) -> None:
        # FIXME: SharedData -> Project -> Pipeline -> SharedData -> ...
        from temporal.preset import Preset
        from temporal.project import Project

        @dataclass
        class ExecutionState:
            # TODO: Probably should not leave API's boundaries
            active_project: Project = field(default_factory = Project)
            # TODO: Probably should belong to an engine?
            running: bool = False
            state: Literal["active", "stopping", "stopped"] = "stopped"
            current_iteration: int = 0
            total_iterations: int = 0
            preview: Optional[NumpyImage] = None

        self.backend = backend
        self.settings_path = settings_path
        self.settings = Settings.load(settings_path)
        self.preset_store = FSStore(Preset, self.settings.fs.preset_dir, self.settings.fs.preset_sorting_order)
        self.preset_store.refresh()
        self.project_store = FSStore(Project, self.settings.fs.project_dir, self.settings.fs.project_sorting_order)
        self.project_store.refresh()
        self.sample_image = pil_to_np(load_image("data/sample_image.png"))
        self.state = ExecutionState()
        self.state_lock = Lock()

        import_modules(list_modules_in_directory("temporal/pipeline_modules", True, 4))
        import_modules(list_modules_in_directory("temporal/video_filters"))


shared = SharedData()
