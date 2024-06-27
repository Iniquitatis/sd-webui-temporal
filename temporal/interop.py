from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

from modules.processing import StableDiffusionProcessing
from modules.scripts import basedir


EXTENSION_DIR = Path(basedir())


@dataclass
class ControlNetUnitWrapper:
    instance: Any = None


@dataclass
class ControlNetUnitList:
    units: list[ControlNetUnitWrapper] = field(default_factory = list)


def import_cn() -> Optional[ModuleType]:
    try:
        from scripts import external_code
    except:
        external_code = None

    return external_code


def get_cn_units(p: StableDiffusionProcessing) -> Optional[ControlNetUnitList]:
    if not (external_code := import_cn()):
        return None

    return ControlNetUnitList([ControlNetUnitWrapper(x) for x in external_code.get_all_units_in_processing(p)])
