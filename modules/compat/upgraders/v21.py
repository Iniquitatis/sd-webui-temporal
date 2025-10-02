import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from modules.compat.upgrader import Upgrader
from modules.utils.fs import load_text, save_text
from modules.utils.image import load_image, pil_to_np
from modules.utils.numpy import save_array


class _(Upgrader):
    version = 21

    def upgrade(self, path: Path) -> bool:
        def convert(elem: Optional[ET.Element]) -> None:
            if elem is None or elem.text is None:
                return

            im_path = path / "project" / "session" / Path(elem.text)

            if not im_path.exists():
                return

            arr_path = im_path.with_suffix(".npz")

            im = load_image(im_path)
            save_array(pil_to_np(im), arr_path)

            elem.set("type", "numpy.ndarray")
            elem.text = arr_path.name

            im_path.unlink()

        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)

        for elem in tree.findall(".//*[@key='mask']/*[@key='image']"):
            convert(elem)

        convert(tree.find(".//*[@key='color_correction']/*[@key='image']"))
        convert(tree.find(".//*[@key='image_overlay']/*[@key='image']"))
        convert(tree.find(".//*[@key='palettization']/*[@key='palette']"))

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")
        save_text(version_path, str(self.version))

        return True
