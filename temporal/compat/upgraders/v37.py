import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader


class _(Upgrader):
    version = 37

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        for base_path in (
            "*[@key='initial_noise']",
            "*[@key='pipeline']/*[@key='modules']/*[@key='noise_overlay']",
        ):
            if (noise := tree.find(f"{base_path}/*[@key='noise']")) is not None:
                ET.SubElement(noise, "object", {"key": "mode", "type": "str"}).text = "fbm"

                if (octaves := noise.find("*[@key='octaves']")) is not None:
                    octaves.set("key", "detail")
                    octaves.set("type", "float")

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
