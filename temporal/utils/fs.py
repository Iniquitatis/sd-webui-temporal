import json
from pathlib import Path
from shutil import rmtree
from typing import Any, IO, Iterator, Optional, TypeVar

T = TypeVar("T")

def open_utf8(path: Path, mode: str) -> IO[Any]:
    return open(path, mode, encoding = "utf-8")

def load_text(path: Path, fallback: T = None) -> str | T:
    path = Path(path)

    if not path.is_file():
        return fallback

    with open_utf8(path, "r") as file:
        return file.read()

def save_text(path: Path, text: str) -> None:
    with open_utf8(path, "w") as file:
        file.write(text)

def load_json(path: Path, fallback: T = None) -> dict[str, Any] | T:
    path = Path(path)

    if not path.is_file():
        return fallback

    with open_utf8(path, "r") as file:
        return json.load(file)

def save_json(path: Path, data: dict[str, Any]) -> None:
    with open_utf8(path, "w") as file:
        json.dump(data, file, indent = 4)

def clear_directory(path: Path, pattern: Optional[str] = None) -> Path:
    path = Path(path)

    if not path.is_dir():
        return path

    for entry in path.iterdir():
        if pattern and not entry.match(pattern):
            continue

        if entry.is_file():
            entry.unlink()
        elif entry.is_dir():
            clear_directory(entry)
            entry.rmdir()

    return path

def ensure_directory_exists(path: Path) -> Path:
    path = Path(path)

    if not path.is_dir():
        path.mkdir(parents = True)

    return path

def is_directory_empty(path: Path) -> bool:
    path = Path(path)

    if not path.is_dir():
        return True

    return sum(1 for _ in path.iterdir()) == 0

def iterate_subdirectories(path: Path) -> Iterator[Path]:
    path = Path(path)

    if not path.is_dir():
        return

    for entry in path.iterdir():
        if entry.is_dir():
            yield entry

def move_entry(old_path: Path, new_path: Path) -> Path:
    old_path = Path(old_path)
    new_path = Path(new_path)

    if not old_path.exists():
        return old_path

    old_path.rename(new_path)

    return new_path

def recreate_directory(path: Path) -> Path:
    path = Path(path)

    remove_directory(path)
    ensure_directory_exists(path)

    return path

def remove_directory(path: Path) -> Path:
    path = Path(path)

    if path.is_dir():
        rmtree(path)

    return path

def remove_entry(path: Path) -> Path:
    path = Path(path)

    if path.is_file():
        path.unlink()
    elif path.is_dir():
        rmtree(path)

    return path

def rename_entry(dir: Path, old_name: str, new_name: str) -> Path:
    return move_entry(dir / old_name, dir / new_name)
