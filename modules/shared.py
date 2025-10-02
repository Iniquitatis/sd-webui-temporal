from pathlib import Path

from modules.backend import Backend
from modules.fs_store import FSStore
from modules.project import Project
from modules.settings import Settings
from modules.utils.image import load_image, pil_to_np
from modules.utils.modules import import_modules, list_modules_in_directory


class SharedData:
    def init(self, backend: Backend, settings_path: Path) -> None:
        self.backend = backend
        self.settings_path = settings_path
        self.settings = Settings.load(settings_path)
        self.project_store = FSStore(Project, self.settings.fs.project_dir, self.settings.fs.project_sorting_order)
        self.project_store.refresh()
        self.sample_image = pil_to_np(load_image("data/sample_image.png"))

        import_modules(list_modules_in_directory("modules/pipeline_modules", True, 4))
        import_modules(list_modules_in_directory("modules/video_filters"))


shared = SharedData()
