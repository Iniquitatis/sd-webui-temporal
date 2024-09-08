from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Optional

from modules.processing import StableDiffusionProcessing


@dataclass
class ControlNetUnitWrapper:
    instance: Any = None


@dataclass
class ControlNetUnitList:
    units: list[ControlNetUnitWrapper] = field(default_factory = list)


def import_controlnet() -> Optional[ModuleType]:
    try:
        from scripts import external_code
    except:
        external_code = None

    return external_code


def get_controlnet_units(p: StableDiffusionProcessing) -> Optional[ControlNetUnitList]:
    if not (external_code := import_controlnet()):
        return None

    return ControlNetUnitList([ControlNetUnitWrapper(x) for x in external_code.get_all_units_in_processing(p)])
