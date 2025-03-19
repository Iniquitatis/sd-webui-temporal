import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader


class _(Upgrader):
    version = 38

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (modules := tree.find("*[@key='pipeline']/*[@key='modules']")) is not None:
            gradient_overlay = ET.SubElement(modules, "object", {"key": "gradient_overlay", "type": "temporal.image_filters.GradientOverlayFilter"})
            ET.SubElement(gradient_overlay, "object", {"key": "enabled", "type": "bool"}).text = "False"
            ET.SubElement(gradient_overlay, "object", {"key": "amount", "type": "float"}).text = "1.0"
            ET.SubElement(gradient_overlay, "object", {"key": "amount_relative", "type": "bool"}).text = "False"
            ET.SubElement(gradient_overlay, "object", {"key": "blend_mode", "type": "str"}).text = "normal"

            mask = ET.SubElement(gradient_overlay, "object", {"key": "mask", "type": "temporal.image_mask.ImageMask"})
            ET.SubElement(mask, "object", {"key": "image", "type": "NoneType"})
            ET.SubElement(mask, "object", {"key": "normalized", "type": "bool"}).text = "False"
            ET.SubElement(mask, "object", {"key": "inverted", "type": "bool"}).text = "False"
            ET.SubElement(mask, "object", {"key": "blurring", "type": "float"}).text = "0.0"

            gradient = ET.SubElement(gradient_overlay, "object", {"key": "gradient", "type": "temporal.gradient.Gradient"})
            ET.SubElement(gradient, "object", {"key": "type", "type": "str"}).text = "linear"
            ET.SubElement(gradient, "object", {"key": "start_x", "type": "float"}).text = "0.0"
            ET.SubElement(gradient, "object", {"key": "start_y", "type": "float"}).text = "0.0"
            ET.SubElement(gradient, "object", {"key": "end_x", "type": "float"}).text = "1.0"
            ET.SubElement(gradient, "object", {"key": "end_y", "type": "float"}).text = "1.0"

            start_color = ET.SubElement(gradient, "object", {"key": "start_color", "type": "temporal.color.Color"})
            ET.SubElement(start_color, "object", {"key": "r", "type": "float"}).text = "1.0"
            ET.SubElement(start_color, "object", {"key": "g", "type": "float"}).text = "1.0"
            ET.SubElement(start_color, "object", {"key": "b", "type": "float"}).text = "1.0"
            ET.SubElement(start_color, "object", {"key": "a", "type": "float"}).text = "1.0"

            end_color = ET.SubElement(gradient, "object", {"key": "end_color", "type": "temporal.color.Color"})
            ET.SubElement(end_color, "object", {"key": "r", "type": "float"}).text = "0.0"
            ET.SubElement(end_color, "object", {"key": "g", "type": "float"}).text = "0.0"
            ET.SubElement(end_color, "object", {"key": "b", "type": "float"}).text = "0.0"
            ET.SubElement(end_color, "object", {"key": "a", "type": "float"}).text = "1.0"

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
