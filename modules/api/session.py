import asyncio
from dataclasses import dataclass, field

from modules.engine import Engine


@dataclass
class Session:
    engine: Engine = field(default_factory = Engine)
    task: asyncio.Task[None] | None = None


global_session = Session()
