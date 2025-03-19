import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader
from temporal.utils.image import load_image, pil_to_np
from temporal.utils.numpy import save_array


class _(Upgrader):
    version = 51

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (images := tree.find("*[@key='parameters']/*[@key='images']")) is not None:
            for image in images:
                if not image.text:
                    continue

                image_path = path / "project" / Path(image.text)
                array_path = image_path.with_suffix(".npz")

                save_array(pil_to_np(load_image(image_path)), array_path)

                image.set("type", "numpy.ndarray")
                image.text = array_path.name

                image_path.unlink()

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
