from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any, Optional

from modules.processing import StableDiffusionProcessing

from temporal.serialization import Serializer


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


class _(Serializer[ControlNetUnitWrapper]):
    keys = [
        "image",
        "enabled",
        "low_vram",
        "pixel_perfect",
        "effective_region_mask",
        "module",
        "model",
        "weight",
        "guidance_start",
        "guidance_end",
        "processor_res",
        "threshold_a",
        "threshold_b",
        "control_mode",
        "resize_mode",
    ]

    @classmethod
    def read(cls, obj, ar):
        for key in cls.keys:
            value = ar[key].create()

            if isinstance(object_value := getattr(obj.instance, key), Enum):
                value = type(object_value)(value)

            setattr(obj.instance, key, value)

        return obj

    @classmethod
    def write(cls, obj, ar):
        for key in cls.keys:
            value = getattr(obj.instance, key)

            if isinstance(value, Enum):
                value = value.value

            ar[key].write(value)


class _(Serializer[ControlNetUnitList]):
    @classmethod
    def read(cls, obj, ar):
        for unit, child in zip(obj.units, ar):
            child.read(unit)

        return obj

    @classmethod
    def write(cls, obj, ar):
        for unit in obj.units:
            ar.make_child().write(unit)
