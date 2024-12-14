import sys
import inspect
import logging
import loguru

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from loguru import Logger, Record


logger: "Logger" = loguru.logger
"""日志记录器对象。

默认信息:

- 格式: `[%(asctime)s %(name)s] %(levelname)s: %(message)s`
- 等级: `INFO` ，根据 `config.log_level` 配置改变
- 输出: 输出至 stdout

"""

log_level = "INFO"

class LoguruHandler(logging.Handler):  # pragma: no cover
    """logging 与 loguru 之间的桥梁，将 logging 的日志转发到 loguru。"""

    def emit(self, record: logging.LogRecord):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def default_filter(record: "Record"):    
    """默认的日志过滤器，根据 `config.log_level` 配置改变日志等级。"""
    levelno = logger.level(log_level).no if isinstance(log_level, str) else log_level
    return record["level"].no >= levelno


def set_log_level(level: str):
    global log_level
    log_level = level

default_format: str = (
    "<g>{time:MM-DD HH:mm:ss}</g> "
    "[<lvl>{level}</lvl>] "
    # "<c><u>{name}</u></c> | "
    # "<c>{function}:{line}</c>| "
    "{message}"
)

logger.remove()
logger_id = logger.add(
    sys.stdout,
    level=0,
    diagnose=False,
    filter=default_filter,
    format=default_format,
)