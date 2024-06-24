from temporal.fs_store import FSStore
from temporal.project import Project


class ProjectStore(FSStore[Project]):
    def __create_entry__(self, name: str) -> Project:
        return Project(self.path / name, name)
