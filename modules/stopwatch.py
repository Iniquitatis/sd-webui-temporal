from statistics import mean
from time import perf_counter
from typing import Any, Self


class Stopwatch:
    def __init__(self, max_samples: int = 100) -> None:
        self._max_samples = max_samples
        self._samples: list[float] = []
        self._start_time = 0.0

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.stop()

    @property
    def last(self) -> float:
        return self._samples[-1] if self._samples else self._elapsed

    @property
    def average(self) -> float:
        return mean(self._samples) if self._samples else self._elapsed

    def start(self) -> None:
        self._start_time = perf_counter()

    def stop(self) -> None:
        self._samples.append(self._elapsed)

        while len(self._samples) > self._max_samples:
            self._samples.pop(0)

    def eta(self, current: int, total: int) -> float:
        remaining = total - current

        if not self._samples:
            return max(0.0, self._elapsed * (remaining - 1))

        return max(0.0, self.average * remaining - self._elapsed)

    def reset(self) -> None:
        self._samples.clear()
        self._start_time = 0.0

    @property
    def _elapsed(self) -> float:
        return perf_counter() - self._start_time if self._start_time > 0.0 else 0.0
