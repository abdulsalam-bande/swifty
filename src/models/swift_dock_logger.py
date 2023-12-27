import logging
import logging.handlers
import os
from logging.handlers import TimedRotatingFileHandler


def swift_dock_logger(log_name="swift_dock.log"):
    logging_format = "[%(asctime)s] %(process)d-%(levelname)s "
    logging_format += "%(message)s"
    log_formatter = logging.Formatter(logging_format)

    logs_dir = "../../logs/"
    os.makedirs(logs_dir, exist_ok=True)

    log_file_name = logs_dir + log_name

    log_handler = TimedRotatingFileHandler(log_file_name, when="midnight")
    log_handler.setFormatter(log_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)

    logger = logging.getLogger(log_file_name)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    logger.addHandler(log_handler)
    logger.addHandler(stream_handler)

    return logger
