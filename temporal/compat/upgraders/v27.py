import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader
from temporal.utils.fs import load_text, move_entry, remove_entry


class _(Upgrader):
    version = 27

    def upgrade(self, path: Path) -> bool:
        version_path = path / "project" / "version.txt"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        session_dir = path / "project" / "session"

        for entry_path in session_dir.iterdir():
            move_entry(entry_path, path / "project" / entry_path.name)

        remove_entry(session_dir)
        remove_entry(version_path)

        new_root = ET.Element("object", {"type": "temporal.project.Project"})
        new_tree = ET.ElementTree(new_root)

        version = ET.SubElement(new_root, "object", {"key": "version", "type": "int"})
        version.text = str(self.version)

        data_path = path / "project" / "data.xml"

        old_tree = ET.ElementTree(file = data_path)

        session = old_tree.getroot()
        session.set("key", "session")
        new_root.append(session)

        ET.indent(new_tree)
        new_tree.write(data_path, "utf-8")

        return True
