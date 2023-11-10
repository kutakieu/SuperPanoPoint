import json
import os
import sys
import traceback
from logging import (
    CRITICAL,
    DEBUG,
    ERROR,
    INFO,
    WARNING,
    Formatter,
    StreamHandler,
    getLogger,
)

log_level_map = {
    "DEBUG": DEBUG,
    "INFO": INFO,
    "WARNING": WARNING,
    "WARN": WARNING,
    "ERROR": ERROR,
    "CRITICAL": CRITICAL,
}
log_level = log_level_map.get(os.environ.get("LOG_LEVEL", "DEBUG").upper(), DEBUG)


class JsonFormatter(Formatter):
    def format(self, record):
        return json.dumps(
            {
                "name": record.name,
                "level": record.levelname,
                "message": record.msg,
                "timestamp": self.formatTime(record, self.datefmt),
                "traceback": traceback.format_exc() if record.exc_info else [],
            },
            ensure_ascii=False,
        )


def get_logger(logger_name: str):
    formatter = JsonFormatter(datefmt="%Y-%m-%d %H:%M:%S")

    stdout_stream = StreamHandler(stream=sys.stdout)
    stdout_stream.setFormatter(formatter)

    stderr_stream = StreamHandler(stream=sys.stderr)
    stderr_stream.setFormatter(formatter)

    logger = getLogger(logger_name)
    logger.setLevel(log_level)
    logger.addHandler(stdout_stream)
    logger.addHandler(stderr_stream)

    return logger
