import asyncio
from dataclasses import dataclass, field

from modules.engine import Engine


@dataclass
class Session:
    engine: Engine = field(default_factory = Engine)
    task: asyncio.Task[None] | None = None

    @property
    def is_task_active(self) -> bool:
        return self.task is not None and not self.task.done()


global_session = Session()
