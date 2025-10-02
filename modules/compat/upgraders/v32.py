import xml.etree.ElementTree as ET
from pathlib import Path

from modules.compat.upgrader import Upgrader


class _(Upgrader):
    version = 32

    def upgrade(self, path: Path) -> bool:
        def update_unit_field(unit: ET.Element, key: str) -> None:
            if (field := unit.find(f"*[@key='instance.{key}']")) is not None:
                field.set("key", key)

        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (controlnet_units := tree.find("*[@key='session']/*[@key='controlnet_units']")) is not None:
            if controlnet_units.get("type", "NoneType") != "NoneType":
                controlnet_units.set("type", "temporal.interop.ControlNetUnitList")

                for unit in controlnet_units:
                    update_unit_field(unit, "enabled")
                    update_unit_field(unit, "module")
                    update_unit_field(unit, "model")
                    update_unit_field(unit, "weight")
                    update_unit_field(unit, "image")
                    update_unit_field(unit, "resize_mode")
                    update_unit_field(unit, "low_vram")
                    update_unit_field(unit, "processor_res")
                    update_unit_field(unit, "threshold_a")
                    update_unit_field(unit, "threshold_b")
                    update_unit_field(unit, "guidance_start")
                    update_unit_field(unit, "guidance_end")
                    update_unit_field(unit, "pixel_perfect")
                    update_unit_field(unit, "control_mode")

                    ET.SubElement(unit, "object", {"key": "effective_region_mask", "type": "NoneType"})

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
