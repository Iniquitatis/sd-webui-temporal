from collections.abc import Iterable
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Iterator, Optional

from temporal import REPO_ROOT


def import_modules(names: Iterable[str]) -> dict[str, ModuleType]:
    return {x: import_module(x) for x in names}


def list_modules_in_directory(path: str | Path, recursive: bool = False, depth: Optional[int] = None) -> Iterator[str]:
    path = Path(path)

    glob = path.rglob if recursive else path.glob

    for subpath in glob("*"):
        if subpath.is_dir() and (subpath / "__init__.py").is_file():
            parts = subpath.relative_to(REPO_ROOT).parts
        elif subpath.is_file() and subpath.suffix == ".py" and subpath.stem != "__init__":
            parts = subpath.relative_to(REPO_ROOT).parent.parts + (subpath.stem,)
        else:
            continue

        if depth is None or len(parts) == depth:
            yield ".".join(parts)
