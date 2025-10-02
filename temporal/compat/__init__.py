from pathlib import Path

from temporal.compat.upgrader import Upgrader
from temporal.utils.logging import log
from temporal.utils.modules import import_modules, list_modules_in_directory


import_modules(list_modules_in_directory("temporal/compat/upgraders"))


def get_latest_version() -> int:
    return max(x.version for x in Upgrader.__subtypes__)


def upgrade_project(path: Path) -> None:
    last_version = 0

    for cls in Upgrader.__subtypes__:
        upgrader = cls()

        if upgrader.upgrade(path):
            last_version = upgrader.version

    if last_version:
        log.info(f"Upgraded project to version {last_version}")
