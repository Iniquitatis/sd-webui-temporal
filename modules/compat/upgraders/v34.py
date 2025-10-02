import xml.etree.ElementTree as ET
from pathlib import Path

from modules.compat.upgrader import Upgrader


class _(Upgrader):
    version = 34

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (pipeline := tree.find("*[@key='session']/*[@key='pipeline']")) is not None:
            actual_module_order: list[str] = []

            if (module_order := pipeline.find("*[@key='module_order']")) is not None:
                for elem in module_order:
                    actual_module_order.append(elem.text or "")

                pipeline.remove(module_order)

            if (modules := pipeline.find("*[@key='modules']")) is not None:
                detached_modules: dict[str, ET.Element] = {}

                for elem in list(modules):
                    detached_modules[elem.get("key", "")] = elem
                    modules.remove(elem)

                for key in actual_module_order:
                    modules.append(detached_modules[key])

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
