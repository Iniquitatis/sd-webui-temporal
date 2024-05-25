from time import sleep
from typing import Callable, Optional


def wait_until(func: Callable[[], bool], interval: int = 1, time_limit: Optional[int] = None) -> None:
    total_time = 0

    while (not func()) and (time_limit is None or total_time < time_limit):
        sleep(interval)

        total_time += interval
