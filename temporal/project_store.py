from pathlib import Path

from temporal.project import Project, make_video_file_name
from temporal.utils.collection import natural_sort
from temporal.utils.fs import iterate_subdirectories, remove_entry, rename_entry


class ProjectStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.project_names = []

    def refresh_projects(self) -> None:
        self.project_names.clear()
        self.project_names.extend(x.name for x in iterate_subdirectories(self.path))
        self.project_names[:] = natural_sort(self.project_names)

    def open_project(self, name: str) -> Project:
        return Project(self.path / name)

    def delete_project(self, name: str) -> None:
        remove_entry(self.path / name)
        remove_entry(self.path / make_video_file_name(name, False))
        remove_entry(self.path / make_video_file_name(name, True))
        self.project_names.remove(name)

    def rename_project(self, old_name: str, new_name: str) -> None:
        rename_entry(self.path, old_name, new_name)
        rename_entry(self.path, make_video_file_name(old_name, False), make_video_file_name(new_name, False))
        rename_entry(self.path, make_video_file_name(old_name, True), make_video_file_name(new_name, True))
        self.project_names[self.project_names.index(old_name)] = new_name
        self.project_names[:] = natural_sort(self.project_names)