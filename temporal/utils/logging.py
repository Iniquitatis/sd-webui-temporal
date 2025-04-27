from datetime import datetime
from enum import IntEnum, auto


class LogLevel(IntEnum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    NO = auto()


log_level = LogLevel.INFO


def log(*values):
    print("[\x1b[97mTEMPORAL\x1b[0m]", f"[\x1b[34m{datetime.now().strftime('%H:%M:%S')}\x1b[0m]", *values)


def debug(*values):
    if log_level <= LogLevel.DEBUG:
        log("[\x1b[90mDEBUG\x1b[0m]", *values)


def info(*values):
    if log_level <= LogLevel.INFO:
        log("[\x1b[36mINFO\x1b[0m]", *values)


def warning(*values):
    if log_level <= LogLevel.WARNING:
        log("[\x1b[33mWARNING\x1b[0m]", *values)


def error(*values):
    if log_level <= LogLevel.ERROR:
        log("[\x1b[31mERROR\x1b[0m]", *values)
