import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader


class _(Upgrader):
    version = 35

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        project = tree.getroot()

        if (session := project.find("*[@key='session']")) is not None:
            for elem in list(session):
                project.append(elem)
                session.remove(elem)

            project.remove(session)

        if (initial_noise := project.find("*[@key='initial_noise']")) is not None:
            initial_noise.set("type", "temporal.project.InitialNoiseParams")

        if (iteration := project.find("*[@key='iteration']")) is not None:
            iteration.set("type", "temporal.project.IterationData")

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
