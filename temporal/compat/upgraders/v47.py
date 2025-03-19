import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader


class _(Upgrader):
    version = 47

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (modules := tree.find("*[@key='pipeline']/*[@key='modules']")) is not None:
            modules.append(ET.fromstring("""
            <object type="temporal.pipeline_modules.measuring.difference.DifferenceMeasuringModule">
              <object key="enabled" type="bool">False</object>
              <object key="plot_every_nth_frame" type="int">10</object>
              <object key="data" type="NoneType" />
              <object key="count" type="int">0</object>
              <object key="last_images" type="list" />
            </object>
            """))

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
