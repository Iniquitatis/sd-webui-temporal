from threading import Lock, Thread

class ThreadQueue:
    def __init__(self):
        self._queue = []
        self._execution_lock = Lock()
        self._queue_lock = Lock()

    @property
    def busy(self):
        with self._queue_lock:
            return len(self._queue) > 0

    def enqueue(self, target, *args, **kwargs):
        def callback():
            with self._execution_lock:
                target(*args, **kwargs)

            with self._queue_lock:
                self._queue.pop(0)

        with self._queue_lock:
            thread = Thread(target = callback)
            self._queue.append(thread)
            thread.start()
