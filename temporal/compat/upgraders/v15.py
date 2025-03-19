from pathlib import Path

from temporal.compat.upgrader import Upgrader
from temporal.utils.fs import ensure_directory_exists, load_text, move_entry, save_text


class _(Upgrader):
    version = 15

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        project_data_dir = ensure_directory_exists(path / "project")
        move_entry(path / "metrics", project_data_dir / "metrics")
        move_entry(path / "session" / "buffer", project_data_dir / "buffer")
        move_entry(path / "session", project_data_dir / "session")
        move_entry(version_path, project_data_dir / "version.txt")

        save_text(project_data_dir / "version.txt", str(self.version))

        return True
