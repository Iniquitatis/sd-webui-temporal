import xml.etree.ElementTree as ET
from pathlib import Path

from modules.compat.upgrader import Upgrader


class _(Upgrader):
    version = 28

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (initial_noise := tree.find("*[@key='session']/*[@key='initial_noise']")) is not None:
            noise = ET.SubElement(initial_noise, "object", {"key": "noise", "type": "temporal.noise.Noise"})

            if (scale := initial_noise.find("*[@key='scale']")) is not None:
                noise.append(scale)
                initial_noise.remove(scale)

            if (octaves := initial_noise.find("*[@key='octaves']")) is not None:
                noise.append(octaves)
                initial_noise.remove(octaves)

            if (lacunarity := initial_noise.find("*[@key='lacunarity']")) is not None:
                noise.append(lacunarity)
                initial_noise.remove(lacunarity)

            if (persistence := initial_noise.find("*[@key='persistence']")) is not None:
                noise.append(persistence)
                initial_noise.remove(persistence)

            ET.SubElement(noise, "object", {"key": "seed", "type": "int"}).text = "0"
            ET.SubElement(initial_noise, "object", {"key": "use_initial_seed", "type": "bool"}).text = "True"

        if (color := tree.find("*[@key='session']/*[@key='pipeline']/*[@key='modules']/*[@key='color_overlay']/*[@key='color']")) is not None:
            color.set("type", "temporal.color.Color")

            if color.text is not None:
                parts = [color.text[i:i + 2] for i in range(1, 7, 2)]

                color.text = None

                ET.SubElement(color, "object", {"key": "r", "type": "float"}).text = str(int(parts[0], 16) / 255.0)
                ET.SubElement(color, "object", {"key": "g", "type": "float"}).text = str(int(parts[1], 16) / 255.0)
                ET.SubElement(color, "object", {"key": "b", "type": "float"}).text = str(int(parts[2], 16) / 255.0)
                ET.SubElement(color, "object", {"key": "a", "type": "float"}).text = "1.0"

        if (noise_overlay := tree.find("*[@key='session']/*[@key='pipeline']/*[@key='modules']/*[@key='noise_overlay']")) is not None:
            noise = ET.SubElement(noise_overlay, "object", {"key": "noise", "type": "temporal.noise.Noise"})

            if (scale := noise_overlay.find("*[@key='scale']")) is not None:
                noise.append(scale)
                noise_overlay.remove(scale)

            if (octaves := noise_overlay.find("*[@key='octaves']")) is not None:
                noise.append(octaves)
                noise_overlay.remove(octaves)

            if (lacunarity := noise_overlay.find("*[@key='lacunarity']")) is not None:
                noise.append(lacunarity)
                noise_overlay.remove(lacunarity)

            if (persistence := noise_overlay.find("*[@key='persistence']")) is not None:
                noise.append(persistence)
                noise_overlay.remove(persistence)

            if (seed := noise_overlay.find("*[@key='seed']")) is not None:
                noise.append(seed)
                noise_overlay.remove(seed)

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
