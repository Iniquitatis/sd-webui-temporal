from pathlib import Path
from typing import Generic, Type, TypeVar, get_args

from temporal.meta.serializable import Serializable
from temporal.utils.collection import natural_sort
from temporal.utils.fs import iterate_subdirectories, remove_entry, rename_entry


T = TypeVar("T", bound = Serializable)


class FSStore(Generic[T]):
    def __init__(self, path: Path, sorting_order: str) -> None:
        self.path = path
        self.sorting_order = sorting_order
        self.entry_names: list[str] = []

    def __create_entry__(self, name: str) -> T:
        raise NotImplementedError

    @property
    def type(self) -> Type[T]:
        return get_args(getattr(self, "__orig_bases__")[0])[0]

    def refresh(self) -> None:
        self.entry_names.clear()
        self.entry_names.extend(x.name for x in iterate_subdirectories(self.path))
        self._sort()

    def load_entry(self, name: str) -> T:
        result = self.__create_entry__(name)
        result.load(self.path / name)
        return result

    def save_entry(self, name: str, entry: T) -> None:
        entry.save(self.path / name)

        if name not in self.entry_names:
            self.entry_names.append(name)
            self._sort()

    def delete_entry(self, name: str) -> None:
        remove_entry(self.path / name)
        self.entry_names.remove(name)

    def rename_entry(self, old_name: str, new_name: str) -> None:
        rename_entry(self.path, old_name, new_name)
        self.entry_names[self.entry_names.index(old_name)] = new_name
        self._sort()

    def _sort(self) -> None:
        if self.sorting_order == "alphabetical":
            self.entry_names[:] = natural_sort(self.entry_names)
        elif self.sorting_order == "date":
            self.entry_names[:] = sorted(self.entry_names, key = lambda x: (self.path / x).stat().st_ctime_ns)
