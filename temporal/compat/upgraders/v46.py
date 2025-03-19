import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader


class _(Upgrader):
    version = 46

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (module := tree.find("*[@key='pipeline']/*[@key='modules']/*[@type='temporal.pipeline_modules.temporal.random_sampling.RandomSamplingModule']")) is not None:
            ET.SubElement(module, "object", {"key": "opacity", "type": "float"}).text = "1.0"

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
