from pathlib import Path

from temporal.project import Project
from temporal.utils.collection import natural_sort
from temporal.utils.fs import iterate_subdirectories, remove_entry, rename_entry


class ProjectStore:
    def __init__(self, path: Path, sorting_order: str) -> None:
        self.path = path
        self.sorting_order = sorting_order
        self.project_names: list[str] = []

    def refresh_projects(self) -> None:
        self.project_names.clear()
        self.project_names.extend(x.name for x in iterate_subdirectories(self.path))
        self._sort()

    def open_project(self, name: str) -> Project:
        project = Project(self.path / name, name)
        project.load(project.path)
        return project

    def delete_project(self, name: str) -> None:
        remove_entry(self.path / name)
        self.project_names.remove(name)

    def rename_project(self, old_name: str, new_name: str) -> None:
        rename_entry(self.path, old_name, new_name)
        self.project_names[self.project_names.index(old_name)] = new_name
        self._sort()

    def _sort(self) -> None:
        if self.sorting_order == "alphabetical":
            self.project_names[:] = natural_sort(self.project_names)
        elif self.sorting_order == "date":
            self.project_names[:] = sorted(self.project_names, key = lambda x: (self.path / x).stat().st_ctime_ns)
