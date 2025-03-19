import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader


class _(Upgrader):
    version = 49

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (modules := tree.find("*[@key='pipeline']/*[@key='modules']")) is not None:
            modules.append(ET.fromstring("""
            <object type="temporal.pipeline_modules.neural.resampling.ResamplingModule">
              <object key="enabled" type="bool">False</object>
              <object key="upscaler" type="str">R-ESRGAN 4x+</object>
              <object key="scale" type="float">1.0</object>
            </object>
            """))

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
