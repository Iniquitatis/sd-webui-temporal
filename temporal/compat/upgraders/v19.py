import xml.etree.ElementTree as ET
from pathlib import Path

from modules.compat.upgrader import Upgrader
from modules.utils.fs import load_text, save_text


class _(Upgrader):
    version = 19

    def upgrade(self, path: Path) -> bool:
        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)
        root = tree.getroot()

        if (modules := root.find(".//*[@key='pipeline']/*[@key='modules']")) is not None:
            limiting = ET.SubElement(modules, "object", {"key": "limiting", "type": "temporal.pipeline_modules.LimitingModule"})
            ET.SubElement(limiting, "object", {"key": "enabled", "type": "bool"}).text = "False"
            ET.SubElement(limiting, "object", {"key": "preview", "type": "bool"}).text = "True"
            ET.SubElement(limiting, "object", {"key": "mode", "type": "str"}).text = "clamp"
            ET.SubElement(limiting, "object", {"key": "max_difference", "type": "float"}).text = "1.0"
            ET.SubElement(limiting, "object", {"key": "buffer", "type": "NoneType"})

        if (module_order := root.find(".//*[@key='pipeline']/*[@key='module_order']")) is not None:
            ET.SubElement(module_order, "object", {"type": "str"}).text = "limiting"

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")
        save_text(version_path, str(self.version))

        return True
