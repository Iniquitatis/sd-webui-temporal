from dataclasses import dataclass, field

from typing import Optional

from temporal.engine import Engine
from temporal.project import Project


@dataclass
class Session:
    engine: Engine = field(default_factory = Engine)
    active_project: Optional[Project] = field(default = None)


global_session = Session()
