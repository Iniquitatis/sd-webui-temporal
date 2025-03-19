import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from temporal.compat.upgrader import Upgrader
from temporal.utils.fs import load_text, save_text
from temporal.utils.numpy import load_array, save_array


class _(Upgrader):
    version = 25

    def upgrade(self, path: Path) -> bool:
        def upgrade_buffer(elem: ET.Element) -> None:
            if elem.text is not None and (arr_path := (path / "project" / "session" / elem.text)).is_file():
                save_array(np.expand_dims(load_array(arr_path), 0), arr_path)

        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)

        if (pipeline := tree.find(".//*[@key='pipeline']")) is not None:
            ET.SubElement(pipeline, "object", {"key": "parallel", "type": "int"}).text = "1"

        if (processing := tree.find(".//*[@key='pipeline']/*[@key='modules']/*[@key='processing']")) is not None:
            ET.SubElement(processing, "object", {"key": "pixels_per_batch", "type": "int"}).text = "1048576"

        if (buffer := tree.find(".//*[@key='pipeline']/*[@key='modules']/*[@key='averaging']/*[@key='buffer']")) is not None:
            upgrade_buffer(buffer)

        if (buffer := tree.find(".//*[@key='pipeline']/*[@key='modules']/*[@key='interpolation']/*[@key='buffer']")) is not None:
            upgrade_buffer(buffer)

        if (buffer := tree.find(".//*[@key='pipeline']/*[@key='modules']/*[@key='limiting']/*[@key='buffer']")) is not None:
            upgrade_buffer(buffer)

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")
        save_text(version_path, str(self.version))

        return True
