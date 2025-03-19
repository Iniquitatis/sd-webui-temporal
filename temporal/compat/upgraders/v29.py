import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader


class _(Upgrader):
    version = 29

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        for module_id in (
            "color_correction",
            "image_overlay",
        ):
            if (module := tree.find(f"*[@key='session']/*[@key='pipeline']/*[@key='modules']/*[@key='{module_id}']")) is not None:
                source = ET.SubElement(module, "object", {"key": "source", "type": "temporal.image_source.ImageSource"})
                ET.SubElement(source, "object", {"key": "type", "type": "str"}).text = "image"

                if (image := module.find("*[@key='image']")) is not None:
                    image.set("key", "value")
                    source.append(image)
                    module.remove(image)

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
