import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader
from temporal.utils import logging
from temporal.utils.fs import load_text, save_text
from temporal.utils.image import load_image, pil_to_np
from temporal.utils.numpy import save_array


class _(Upgrader):
    version = 24

    def upgrade(self, path: Path) -> bool:
        def parse_frame_index(im_path: Path) -> int:
            if im_path.is_file():
                try:
                    return int(im_path.stem)
                except:
                    logging.warning(f"{im_path.stem} doesn't match the frame name format")

            return 0

        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)
        root = tree.getroot()

        last_frame_index = max((parse_frame_index(x) for x in path.glob("*.png")), default = 0)

        iteration = ET.SubElement(root, "object", {"key": "iteration", "type": "temporal.session.IterationData"})

        images = ET.SubElement(iteration, "object", {"key": "images", "type": "list"})

        im_path = None

        if (last_frame_path := (path / f"{last_frame_index:05d}.png")).is_file():
            im_path = last_frame_path
        elif (init_image_name := tree.findtext("*[@key='processing']/*[@key='init_images']/*[@type='PIL.Image.Image']")) is not None:
            im_path = path / "project" / "session" / init_image_name

        if im_path is not None:
            arr_path = path / "project" / "session" / "last_frame.npz"

            im = load_image(im_path)
            save_array(pil_to_np(im), arr_path)

            image = ET.SubElement(images, "object", {"type": "numpy.ndarray"})
            image.text = arr_path.name

        index = ET.SubElement(iteration, "object", {"key": "index", "type": "int"})
        index.text = str(last_frame_index + 1)

        ET.SubElement(iteration, "object", {"key": "module_id", "type": "NoneType"})

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")
        save_text(version_path, str(self.version))

        return True
