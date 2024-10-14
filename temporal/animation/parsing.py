import re
from dataclasses import dataclass, field
from typing import Any, cast

from temporal.animation import Animation, Track
from temporal.color import Color


def parse_animation(code: str) -> Animation:
    return _read_animation(_parse_tree(code))


@dataclass
class Node:
    level: int = -1
    key: str = ""
    arg: str = ""
    value: str = ""
    children: list["Node"] = field(default_factory = list)


def _parse_tree(code: str, indentation: int = 4) -> Node:
    result = Node(key = "root")
    parent_stack = [result]
    last_node = result

    for line in code.splitlines():
        if not (m := re.match(r"(\s*)(\w+)(?:\s+(.+?))?\s*:\s*(?:(.*\S))?", line)):
            continue

        spaces, key, arg, value = m.groups()

        new_node = Node()
        new_node.level = len(spaces) // indentation
        new_node.key = key
        new_node.arg = arg
        new_node.value = value

        if new_node.level > last_node.level:
            parent_stack.append(last_node)

        for _ in range(max(last_node.level - new_node.level, 0)):
            parent_stack.pop()

        parent_node = parent_stack[-1]
        parent_node.children.append(new_node)

        last_node = new_node

    return result


def _read_animation(node: Node) -> Animation:
    result = Animation()

    for child in node.children:
        if child.key == "track":
            result.tracks[child.arg] = _read_track(child)

    return result


def _read_track(node: Node) -> Track:
    result = Track()

    for child in node.children:
        if child.key == "interpolation":
            result.interpolation = cast(Any, child.value)
        elif child.key == "bounds":
            result.bounds = cast(Any, child.value)
        elif child.key == "frame":
            result.add_keyframe(int(child.arg), eval(child.value, {"Color": Color}))

    return result
