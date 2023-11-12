from pathlib import Path

from modules import scripts

EXTENSION_DIR = Path(scripts.basedir())

def import_cn():
    try:
        from scripts import external_code
    except:
        external_code = None

    return external_code
