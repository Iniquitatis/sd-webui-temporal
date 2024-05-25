from pathlib import Path
from types import ModuleType
from typing import Any, Optional

from modules import scripts
from modules.processing import StableDiffusionProcessing

EXTENSION_DIR = Path(scripts.basedir())

def import_cn() -> Optional[ModuleType]:
    try:
        from scripts import external_code
    except:
        external_code = None

    return external_code

def get_cn_units(p: StableDiffusionProcessing) -> Optional[list[Any]]:
    if not (external_code := import_cn()):
        return None

    return external_code.get_all_units_in_processing(p)
