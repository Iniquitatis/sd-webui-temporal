from importlib import import_module
from pathlib import Path


for path in (Path(__file__).parent / "pipeline_modules").rglob("*.py"):
    if path.name != "__init__":
        import_module(f"temporal.pipeline_modules.{path.parent.stem}.{path.stem}")
