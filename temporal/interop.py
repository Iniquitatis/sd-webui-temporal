from pathlib import Path

from modules import scripts

EXTENSION_DIR = Path(scripts.basedir())

def import_cn():
    try:
        from scripts import external_code
    except:
        external_code = None

    return external_code

def get_cn_units(p):
    if not (external_code := import_cn()):
        return None

    return external_code.get_all_units_in_processing(p)
