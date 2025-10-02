import xml.etree.ElementTree as ET
from pathlib import Path

from modules.compat.upgrader import Upgrader
from modules.utils.fs import load_text, save_text


class _(Upgrader):
    version = 23

    def upgrade(self, path: Path) -> bool:
        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)

        if (frame_merging := tree.find(".//*[@key='frame_merging']")) is not None:
            frame_merging.set("key", "averaging")
            frame_merging.set("type", "temporal.pipeline_modules.AveragingModule")

        if (dampening := tree.find(".//*[@key='dampening']")) is not None:
            dampening.set("key", "interpolation")
            dampening.set("type", "temporal.pipeline_modules.InterpolationModule")

        if (frame_merging_id := tree.find(".//*[@key='pipeline']/*[@key='module_order']/*[.='frame_merging']")) is not None:
            frame_merging_id.text = "averaging"

        if (dampening_id := tree.find(".//*[@key='pipeline']/*[@key='module_order']/*[.='dampening']")) is not None:
            dampening_id.text = "interpolation"

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")
        save_text(version_path, str(self.version))

        return True
