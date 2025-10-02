import xml.etree.ElementTree as ET
from pathlib import Path

from modules.compat.upgrader import Upgrader
from modules.utils.fs import load_text, save_text


class _(Upgrader):
    version = 26

    def upgrade(self, path: Path) -> bool:
        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)

        if (interpolation := tree.find(".//*[@key='pipeline']/*[@key='modules']/*[@key='interpolation']")) is not None:
            if (rate := interpolation.find("*[@key='rate']")) is not None:
                rate.set("key", "blending")

            ET.SubElement(interpolation, "object", {"key": "movement", "type": "float"}).text = "0.0"
            ET.SubElement(interpolation, "object", {"key": "radius", "type": "int"}).text = "15"

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")
        save_text(version_path, str(self.version))

        return True
