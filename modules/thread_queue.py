from threading import Lock, Thread
from traceback import format_exc
from typing import Any, Callable, ParamSpec

from modules.utils.logging import log


P = ParamSpec("P")


class ThreadQueue:
    def __init__(self) -> None:
        self._queue = []
        self._execution_lock = Lock()
        self._queue_lock = Lock()

    @property
    def busy(self) -> bool:
        with self._queue_lock:
            return len(self._queue) > 0

    def enqueue(self, target: Callable[P, Any], *args: P.args, **kwargs: P.kwargs) -> None:
        def callback() -> None:
            with self._execution_lock:
                try:
                    target(*args, **kwargs)
                except:
                    log.error(f"Exception from queued thread:\n{format_exc()}")

            with self._queue_lock:
                self._queue.pop(0)

        with self._queue_lock:
            thread = Thread(target = callback)
            self._queue.append(thread)
            thread.start()
