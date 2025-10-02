from dataclasses import dataclass, field

from modules.engine import Engine


@dataclass
class Session:
    engine: Engine = field(default_factory = Engine)


global_session = Session()
