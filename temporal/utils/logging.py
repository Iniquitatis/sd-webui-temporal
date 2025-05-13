from datetime import datetime
from enum import IntEnum, auto
from pathlib import Path
from sys import stdout
from threading import Lock
from typing import Any, IO, Optional


class LogLevel(IntEnum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    NO = auto()


class Logger:
    def __init__(self) -> None:
        self.level = LogLevel.INFO
        self._file: Optional[IO[str]] = None
        self._lock = Lock()

    @property
    def path(self) -> None:
        pass

    @path.setter
    def path(self, value: Optional[str | Path]) -> None:
        if self._file is not None:
            self._file.close()

        if isinstance(value, (str, Path)):
            self._file = open(value, "w", encoding = "utf-8")

    def debug(self, *values: Any) -> None:
        if self.level <= LogLevel.DEBUG:
            self._print("DEBUG", 90, *values)

    def info(self, *values: Any) -> None:
        if self.level <= LogLevel.INFO:
            self._print("INFO", 36, *values)

    def warning(self, *values: Any) -> None:
        if self.level <= LogLevel.WARNING:
            self._print("WARNING", 33, *values)

    def error(self, *values: Any) -> None:
        if self.level <= LogLevel.ERROR:
            self._print("ERROR", 31, *values)

    def _print(self, prefix: str, color_index: int, *values: Any) -> None:
        self._print_to_stream(stdout, prefix, color_index, True, *values)

        if self._file is not None:
            self._print_to_stream(self._file, prefix, color_index, False, *values)

    def _print_to_stream(
        self,
        stream: IO[str],
        prefix: str,
        color_index: int,
        decorated: bool,
        *values: Any,
    ) -> None:
        def color(text: str, color_index: int, decorated: bool) -> str:
            return f"\x1b[{color_index}m{text}\x1b[0m" if decorated else text

        with self._lock:
            print(
                f"[{color('TEMPORAL', 97, decorated)}]",
                f"[{color(datetime.now().strftime('%H:%M:%S'), 34, decorated)}]",
                f"[{color(prefix, color_index, decorated)}]",
                *values,
                file = stream,
            )


log = Logger()
