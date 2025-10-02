from abc import abstractmethod
from pathlib import Path

from modules.object import Object, Static


class Upgrader(Object, abstract = True):
    version: int = Static(-1)

    @property
    def previous_version(self) -> int:
        return max(x.version for x in Upgrader.__subtypes__ if x.version < self.version)

    @abstractmethod
    def upgrade(self, path: Path) -> bool:
        raise NotImplementedError
