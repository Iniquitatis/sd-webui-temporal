from pathlib import Path
from shutil import copy2

from modules.compat.upgrader import Upgrader
from modules.utils.fs import ensure_directory_exists, load_text, save_text


class _(Upgrader):
    version = 3

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        if frames := sorted(path.glob("*.png"), key = lambda x: int(x.stem)):
            copy2(frames[-1], ensure_directory_exists(path / "session" / "buffer") / "001.png")

        save_text(version_path, str(self.version))

        return True
