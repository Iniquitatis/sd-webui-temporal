import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader


class _(Upgrader):
    version = 33

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (module_order := tree.find("*[@key='session']/*[@key='pipeline']/*[@key='module_order']")) is not None:
            ET.SubElement(module_order, "object", {"type": "str"}).text = "pixelization"

        if (modules := tree.find("*[@key='session']/*[@key='pipeline']/*[@key='modules']")) is not None:
            pixelization = ET.SubElement(modules, "object", {"key": "pixelization", "type": "temporal.image_filters.PixelizationFilter"})
            ET.SubElement(pixelization, "object", {"key": "enabled", "type": "bool"}).text = "False"
            ET.SubElement(pixelization, "object", {"key": "amount", "type": "float"}).text = "1.0"
            ET.SubElement(pixelization, "object", {"key": "amount_relative", "type": "bool"}).text = "False"
            ET.SubElement(pixelization, "object", {"key": "blend_mode", "type": "str"}).text = "normal"
            mask = ET.SubElement(pixelization, "object", {"key": "mask", "type": "temporal.image_mask.ImageMask"})
            ET.SubElement(mask, "object", {"key": "image", "type": "NoneType"})
            ET.SubElement(mask, "object", {"key": "normalized", "type": "bool"}).text = "False"
            ET.SubElement(mask, "object", {"key": "inverted", "type": "bool"}).text = "False"
            ET.SubElement(mask, "object", {"key": "blurring", "type": "float"}).text = "0.0"
            ET.SubElement(pixelization, "object", {"key": "pixel_size", "type": "int"}).text = "1"

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
