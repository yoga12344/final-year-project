"""
Structured logging utilities for DR-TBAC-ZT++.

Sets up a consistent logging configuration across all modules:
- Console handler with colour-coded log levels
- Rotating file handler writing to logs/dr_tbac.log
- JSON formatter for machine-readable log ingestion (e.g. ELK stack)
"""

import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_DIR  = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, "dr_tbac.log")
LOG_MAX_BYTES = 10 * 1024 * 1024   # 10 MB
LOG_BACKUP_COUNT = 5


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class ColourFormatter(logging.Formatter):
    """ANSI colour codes for console output."""

    GREY    = "\x1b[38;20m"
    CYAN    = "\x1b[36;20m"
    YELLOW  = "\x1b[33;20m"
    RED     = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET   = "\x1b[0m"

    LEVEL_COLOURS = {
        logging.DEBUG:    GREY,
        logging.INFO:     CYAN,
        logging.WARNING:  YELLOW,
        logging.ERROR:    RED,
        logging.CRITICAL: BOLD_RED,
    }

    _FMT = "[{asctime}] [{levelname:<8}] {name}: {message}"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.LEVEL_COLOURS.get(record.levelno, self.RESET)
        fmt = logging.Formatter(
            f"{colour}{self._FMT}{self.RESET}", style="{"
        )
        return fmt.format(record)


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
            "module":  record.module,
            "lineno":  record.lineno,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Setup function
# ---------------------------------------------------------------------------

def setup_logging(
    *,
    level: int = logging.INFO,
    log_file: str = LOG_FILE,
    json_file_log: bool = True,
) -> logging.Logger:
    """
    Configure the root logger once for the entire application.

    Call this **once** at application start-up (e.g. in ``main.py``).

    Parameters
    ----------
    level         : int   Root log level (default INFO).
    log_file      : str   Path to the rotating log file.
    json_file_log : bool  If True, write JSON lines to file; else plain text.

    Returns
    -------
    Root ``logging.Logger`` instance.
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers on re-import
    if root.handlers:
        return root

    # ---- Console handler ---------------------------------------------------
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(ColourFormatter())
    root.addHandler(console)

    # ---- Rotating file handler --------------------------------------------
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)   # capture everything to file
    file_handler.setFormatter(
        JSONFormatter() if json_file_log
        else logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(file_handler)

    # Silence noisy third-party libraries
    for noisy in ("urllib3", "grpc", "hfc", "asyncio", "flower"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    """
    Convenience wrapper to get a named child logger.

    Ensures that ``setup_logging()`` has been called first.
    """
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name)
