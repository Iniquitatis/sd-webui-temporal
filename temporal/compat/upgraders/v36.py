import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader


class _(Upgrader):
    version = 36

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (initial_noise := tree.find("*[@key='initial_noise']")) is not None:
            if (use_initial_seed := initial_noise.find("*[@key='use_initial_seed']")) is not None:
                if (noise := initial_noise.find("*[@key='noise']")) is not None:
                    ET.SubElement(noise, "object", {"key": "use_global_seed", "type": "bool"}).text = use_initial_seed.text

                initial_noise.remove(use_initial_seed)

        if (noise_overlay := tree.find("*[@key='pipeline']/*[@key='modules']/*[@key='noise_overlay']")) is not None:
            if (use_dynamic_seed := noise_overlay.find("*[@key='use_dynamic_seed']")) is not None:
                if (noise := noise_overlay.find("*[@key='noise']")) is not None:
                    ET.SubElement(noise, "object", {"key": "use_global_seed", "type": "bool"}).text = use_dynamic_seed.text

                noise_overlay.remove(use_dynamic_seed)

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
