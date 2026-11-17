import logging
from datetime import datetime
from pathlib import Path

from app.core.config import BASE_DIR

LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_DIR / f"log_{datetime.now().strftime('%Y-%m-%d')}.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# uvicorn configures logging before importing the app, so logging.basicConfig()
# is usually a no-op here. Attach the file handler explicitly instead.
_root = logging.getLogger()
_root.setLevel(logging.INFO)

_has_file_handler = any(
    isinstance(handler, logging.FileHandler)
    and Path(handler.baseFilename).resolve() == LOG_FILE.resolve()
    for handler in _root.handlers
)

if not _has_file_handler:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    _root.addHandler(file_handler)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
