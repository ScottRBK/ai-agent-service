import logging
import os
import sys
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

# Ensure logs do not propagate to root logger (which might have stdout handlers)
logger.propagate = False

# Route console logs explicitly to stderr to avoid interleaving with CLI stdout streaming
console_handler = logging.StreamHandler(stream=sys.stderr)
console_handler.setLevel(log_level_map.get(LOG_LEVEL, logging.INFO))
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

# Avoid duplicate handlers if this module is imported multiple times
_existing_streams = {getattr(h, 'stream', None) for h in logger.handlers if isinstance(h, logging.StreamHandler)}
if sys.stderr not in _existing_streams:
    logger.addHandler(console_handler)

#TODO: Add file handler