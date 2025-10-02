import xml.etree.ElementTree as ET
from pathlib import Path

from modules.compat.upgrader import Upgrader


class _(Upgrader):
    version = 31

    def upgrade(self, path: Path) -> bool:
        def create_after(parent: ET.Element, sibling: ET.Element, tag: str, attrs: dict[str, str], text: str) -> ET.Element:
            elem = ET.Element(tag, attrs)
            elem.text = text
            parent.insert(list(parent).index(sibling) + 1, elem)
            return elem

        def create_module(modules: ET.Element, key: str, type: str, enabled: bool, plot_every_nth_frame: int) -> None:
            elem = ET.SubElement(modules, "object", {"key": key, "type": type})
            ET.SubElement(elem, "object", {"key": "enabled", "type": "bool"}).text = str(enabled)
            ET.SubElement(elem, "object", {"key": "plot_every_nth_frame", "type": "int"}).text = str(plot_every_nth_frame)
            ET.SubElement(elem, "object", {"key": "data", "type": "NoneType"})
            ET.SubElement(elem, "object", {"key": "count", "type": "int"}).text = "0"

        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (module_order := tree.find("*[@key='session']/*[@key='pipeline']/*[@key='module_order']")) is not None:
            if (measuring := module_order.find("*[.='measuring']")) is not None:
                elem = measuring

                elem = create_after(module_order, elem, "object", {"type": "str"}, "color_level_mean_measuring")
                elem = create_after(module_order, elem, "object", {"type": "str"}, "color_level_sigma_measuring")
                elem = create_after(module_order, elem, "object", {"type": "str"}, "luminance_mean_measuring")
                elem = create_after(module_order, elem, "object", {"type": "str"}, "luminance_sigma_measuring")
                elem = create_after(module_order, elem, "object", {"type": "str"}, "noise_sigma_measuring")

                module_order.remove(measuring)

        if (modules := tree.find("*[@key='session']/*[@key='pipeline']/*[@key='modules']")) is not None:
            if (measuring := modules.find("*[@key='measuring']")) is not None:
                enabled = measuring.findtext("*[@key='enabled']", "False") == "True"
                plot_every_nth_frame = int(measuring.findtext("*[@key='plot_every_nth_frame']", "10"))

                create_module(modules, "color_level_mean_measuring", "temporal.measuring_modules.ColorLevelMeanMeasuringModule", enabled, plot_every_nth_frame)
                create_module(modules, "color_level_sigma_measuring", "temporal.measuring_modules.ColorLevelSigmaMeasuringModule", enabled, plot_every_nth_frame)
                create_module(modules, "luminance_mean_measuring", "temporal.measuring_modules.LuminanceMeanMeasuringModule", enabled, plot_every_nth_frame)
                create_module(modules, "luminance_sigma_measuring", "temporal.measuring_modules.LuminanceSigmaMeasuringModule", enabled, plot_every_nth_frame)
                create_module(modules, "noise_sigma_measuring", "temporal.measuring_modules.NoiseSigmaMeasuringModule", enabled, plot_every_nth_frame)

                modules.remove(measuring)

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
