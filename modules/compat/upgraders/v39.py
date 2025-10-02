import xml.etree.ElementTree as ET
from pathlib import Path

from modules.compat.upgrader import Upgrader


class _(Upgrader):
    version = 39

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (modules := tree.find("*[@key='pipeline']/*[@key='modules']")) is not None:
            pattern_overlay = ET.SubElement(modules, "object", {"key": "pattern_overlay", "type": "temporal.image_filters.PatternOverlayFilter"})
            ET.SubElement(pattern_overlay, "object", {"key": "enabled", "type": "bool"}).text = "False"
            ET.SubElement(pattern_overlay, "object", {"key": "amount", "type": "float"}).text = "1.0"
            ET.SubElement(pattern_overlay, "object", {"key": "amount_relative", "type": "bool"}).text = "False"
            ET.SubElement(pattern_overlay, "object", {"key": "blend_mode", "type": "str"}).text = "normal"

            mask = ET.SubElement(pattern_overlay, "object", {"key": "mask", "type": "temporal.image_mask.ImageMask"})
            ET.SubElement(mask, "object", {"key": "image", "type": "NoneType"})
            ET.SubElement(mask, "object", {"key": "normalized", "type": "bool"}).text = "False"
            ET.SubElement(mask, "object", {"key": "inverted", "type": "bool"}).text = "False"
            ET.SubElement(mask, "object", {"key": "blurring", "type": "float"}).text = "0.0"

            pattern = ET.SubElement(pattern_overlay, "object", {"key": "pattern", "type": "temporal.pattern.Pattern"})
            ET.SubElement(pattern, "object", {"key": "type", "type": "str"}).text = "horizontal_lines"
            ET.SubElement(pattern, "object", {"key": "size", "type": "int"}).text = "8"

            color_a = ET.SubElement(pattern, "object", {"key": "color_a", "type": "temporal.color.Color"})
            ET.SubElement(color_a, "object", {"key": "r", "type": "float"}).text = "1.0"
            ET.SubElement(color_a, "object", {"key": "g", "type": "float"}).text = "1.0"
            ET.SubElement(color_a, "object", {"key": "b", "type": "float"}).text = "1.0"
            ET.SubElement(color_a, "object", {"key": "a", "type": "float"}).text = "1.0"

            color_b = ET.SubElement(pattern, "object", {"key": "color_b", "type": "temporal.color.Color"})
            ET.SubElement(color_b, "object", {"key": "r", "type": "float"}).text = "0.0"
            ET.SubElement(color_b, "object", {"key": "g", "type": "float"}).text = "0.0"
            ET.SubElement(color_b, "object", {"key": "b", "type": "float"}).text = "0.0"
            ET.SubElement(color_b, "object", {"key": "a", "type": "float"}).text = "1.0"

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
