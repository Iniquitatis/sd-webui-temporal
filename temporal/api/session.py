from dataclasses import dataclass, field

from temporal.engine import Engine


@dataclass
class Session:
    engine: Engine = field(default_factory = Engine)


global_session = Session()
