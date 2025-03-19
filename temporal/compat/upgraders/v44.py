import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader


class _(Upgrader):
    version = 44

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (modules := tree.find("*[@key='pipeline']/*[@key='modules']")) is not None:
            modules.append(ET.fromstring("""
            <object type="temporal.pipeline_modules.filtering.color_matrix.ColorMatrixFilter">
              <object key="enabled" type="bool">False</object>
              <object key="amount" type="float">1.0</object>
              <object key="amount_relative" type="bool">False</object>
              <object key="blend_mode" type="temporal.blend_modes.NormalBlendMode" />
              <object key="mask" type="temporal.image_mask.ImageMask">
                <object key="image" type="NoneType" />
                <object key="normalized" type="bool">False</object>
                <object key="inverted" type="bool">False</object>
                <object key="blurring" type="float">0.0</object>
              </object>
              <object key="r" type="temporal.color.Color">
                <object key="r" type="float">1.0</object>
                <object key="g" type="float">0.0</object>
                <object key="b" type="float">0.0</object>
                <object key="a" type="float">1.0</object>
              </object>
              <object key="g" type="temporal.color.Color">
                <object key="r" type="float">0.0</object>
                <object key="g" type="float">1.0</object>
                <object key="b" type="float">0.0</object>
                <object key="a" type="float">1.0</object>
              </object>
              <object key="b" type="temporal.color.Color">
                <object key="r" type="float">0.0</object>
                <object key="g" type="float">0.0</object>
                <object key="b" type="float">1.0</object>
                <object key="a" type="float">1.0</object>
              </object>
              <object key="normalized" type="bool">False</object>
            </object>
            """))

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
