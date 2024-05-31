from enum import IntEnum, auto


class LogLevel(IntEnum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    NO = auto()


log_level = LogLevel.INFO


def log(*values):
    print("[Temporal]", *values)


def debug(*values):
    if log_level <= LogLevel.DEBUG:
        log("[DEBUG]", *values)


def info(*values):
    if log_level <= LogLevel.INFO:
        log("[INFO]", *values)


def warning(*values):
    if log_level <= LogLevel.WARNING:
        log("[WARNING]", *values)


def error(*values):
    if log_level <= LogLevel.ERROR:
        log("[ERROR]", *values)
