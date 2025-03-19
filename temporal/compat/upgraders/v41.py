import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader


class _(Upgrader):
    version = 41

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        classes = {
            "normal": "NormalBlendMode",
            "add": "AddBlendMode",
            "subtract": "SubtractBlendMode",
            "multiply": "MultiplyBlendMode",
            "divide": "DivideBlendMode",
            "lighten": "LightenBlendMode",
            "darken": "DarkenBlendMode",
            "hard_light": "HardLightBlendMode",
            "soft_light": "SoftLightBlendMode",
            "color_dodge": "ColorDodgeBlendMode",
            "color_burn": "ColorBurnBlendMode",
            "overlay": "OverlayBlendMode",
            "screen": "ScreenBlendMode",
            "difference": "DifferenceBlendMode",
            "exclusion": "ExclusionBlendMode",
            "hue": "HueBlendMode",
            "saturation": "SaturationBlendMode",
            "value": "ValueBlendMode",
            "color": "ColorBlendMode",
        }

        for blend_mode in tree.iterfind("*[@key='pipeline']/*[@key='modules']/*/*[@key='blend_mode']"):
            if blend_mode.text is not None:
                blend_mode.set("type", f"temporal.blend_modes.{classes[blend_mode.text]}")
                blend_mode.text = None

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
