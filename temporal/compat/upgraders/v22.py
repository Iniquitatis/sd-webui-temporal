import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader
from temporal.utils.fs import load_text, save_text


class _(Upgrader):
    version = 22

    def upgrade(self, path: Path) -> bool:
        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)

        if (module_order := tree.find(".//*[@key='module_order']")) is not None and \
           (filter_order := tree.find(".//*[@key='filter_order']")) is not None:
            if (image_filtering_id := module_order.find(".//*[.='image_filtering']")) is not None:
                image_filtering_index = list(module_order).index(image_filtering_id)

                for i, filter_id in enumerate(filter_order):
                    module_order.insert(image_filtering_index + i, filter_id)

                module_order.remove(image_filtering_id)

            else:
                for filter_id in filter_order:
                    module_order.append(filter_id)

        if (modules := tree.find(".//*[@key='pipeline']/*[@key='modules']")) is not None and \
           (filters := tree.find(".//*[@key='image_filterer']/*[@key='filters']")) is not None:
            for filter in filters:
                modules.append(filter)

            if (image_filtering := modules.find(".//*[@key='image_filtering']")) is not None:
                modules.remove(image_filtering)

        if (image_filterer := tree.find(".//*[@key='image_filterer']")) is not None:
            tree.getroot().remove(image_filterer)

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")
        save_text(version_path, str(self.version))

        return True
