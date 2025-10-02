import xml.etree.ElementTree as ET
from pathlib import Path

from modules.compat.upgrader import Upgrader


class _(Upgrader):
    version = 30

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (module_order := tree.find("*[@key='session']/*[@key='pipeline']/*[@key='module_order']")) is not None:
            ET.SubElement(module_order, "object", {"type": "str"}).text = "random_sampling"

        if (modules := tree.find("*[@key='session']/*[@key='pipeline']/*[@key='modules']")) is not None:
            random_sampling = ET.SubElement(modules, "object", {"key": "random_sampling", "type": "temporal.pipeline_modules.RandomSamplingModule"})
            ET.SubElement(random_sampling, "object", {"key": "enabled", "type": "bool"}).text = "False"
            ET.SubElement(random_sampling, "object", {"key": "chance", "type": "float"}).text = "1.0"
            ET.SubElement(random_sampling, "object", {"key": "buffer", "type": "NoneType"})

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
