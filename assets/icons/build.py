#
# NOTE: Run using "fontforge -script build.py"
#

import tempfile
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen
from typing import Iterator
from xml.etree import ElementTree as ET

import fontforge


input_path = "icons.svg"
output_path = "icons.ttf"
preview_path = "preview.html"


class Icon:
    def __init__(self, id: str) -> None:
        self.id = id
        self.codepoint, self.name = id.split("-", 1)

    @property
    def codepoint_int(self) -> int:
        return int(self.codepoint, 16)

    @property
    def file_name(self) -> str:
        return f"{self.id}.svg"


icons = list(
    Icon(elem.attrib["id"])
    for elem in ET.parse(input_path).getroot().iterfind("./{*}g")
)
icons.reverse()


with tempfile.TemporaryDirectory() as tmp:
    def make_actions_for_one_icon(target: Icon) -> Iterator[str]:
        yield f"file-open:{input_path}"

        for icon in icons:
            if icon.id != target.id:
                yield f"select-by-id:{icon.id}"
                yield f"delete-selection"

        yield f"select-by-id:{target.id}"
        yield f"path-flatten"
        yield f"path-union"

        yield f"export-overwrite"
        yield f"export-filename:{(Path(tmp) / target.file_name).as_posix()}"
        yield f"export-plain-svg"
        yield f"export-do"

    def make_actions() -> Iterator[str]:
        for icon in icons:
            yield from make_actions_for_one_icon(icon)

    proc = Popen("inkscape --shell", stdin = PIPE, stdout = PIPE, stderr = STDOUT, shell = True)
    proc.communicate(input = ";\n".join(make_actions()).encode(), timeout = 600)
    proc.wait()

    font = fontforge.font()
    font.fontname = "Icons"
    font.em = 2048

    for icon in icons:
        glyph = font.createChar(icon.codepoint_int)
        glyph.importOutlines((Path(tmp) / icon.file_name).as_posix(), scale = True, simplify = False, accuracy = 1.0 / 16.0)
        glyph.width = font.em

    font.generate(output_path, flags = ("no-hints", "no-flex", "omit-instructions"))


with open(preview_path, "w") as file:
    def write_code() -> Iterator[str]:
        yield f'<html>'
        yield f'    <head>'
        yield f'        <meta name="viewport" content="width=device-width, initial-scale=1">'
        yield f'        <style>'
        yield f'            @font-face {{ font-family: "Icons"; src: url("{output_path}") format("opentype"); }}'
        yield f'            body {{ display: flex; flex-direction: column; font-family: monospace; gap: 1rem; }}'
        yield f'            .block {{ display: flex; flex-direction: row; gap: 0.5rem; }}'
        yield f'            .glyph {{ color: #101010; font: 16px "Icons"; }}'
        yield f'            .codepoint {{ color: #808080; font-size: 0.9rem; }}'
        yield f'        </style>'
        yield f'    </head>'
        yield f'    <body>'

        for icon in icons:
            yield f'        <div class="block">'
            yield f'            <div class="glyph">&#x{icon.codepoint};</div>'
            yield f'            <div class="codepoint">({icon.codepoint})</div>'
            yield f'            <div>{icon.name}</div>'
            yield f'        </div>'

        yield f'    </body>'
        yield f'</html>'

    file.write("\n".join(write_code()))
