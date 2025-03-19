import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader


class _(Upgrader):
    version = 52

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        tree.getroot().append(ET.fromstring("""
        <object key="animation" type="temporal.animation.Animation" />
        """))

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
