import logging
import os
from app.config.settings import settings

log_level_map = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


LOG_LEVEL = settings.LOG_LEVEL
logger = logging.getLogger("AGENT-SERVICE-LOGGER")
logger.setLevel(log_level_map.get(LOG_LEVEL, logging.INFO))

console_handler = logging.StreamHandler()
console_handler.setLevel(log_level_map.get(LOG_LEVEL, logging.INFO))
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

logger.addHandler(console_handler)

#TODO: Add file handler