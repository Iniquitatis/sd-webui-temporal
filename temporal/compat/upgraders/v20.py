import xml.etree.ElementTree as ET
from pathlib import Path

from modules.compat.upgrader import Upgrader
from modules.utils.fs import load_text, save_text


class _(Upgrader):
    version = 20

    def upgrade(self, path: Path) -> bool:
        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)

        if (frame_merging := tree.find(".//*[@key='frame_merging']")) is not None:
            if (buffer := frame_merging.find("*[@key='buffer']")) is not None:
                for elem in list(buffer):
                    frame_merging.append(elem)
                    buffer.remove(elem)

                frame_merging.remove(buffer)

            if (array := frame_merging.find("*[@key='array']")) is not None:
                array.set("key", "buffer")

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")
        save_text(version_path, str(self.version))

        return True
