import logging
import sys


class ColoredFormatter(logging.Formatter):
    """Colors the level name and message by log level (same scheme as FISH-ML2)."""
    COLORS = {
        "DEBUG": "\033[94m",     # blue
        "INFO": "\033[92m",      # green
        "WARNING": "\033[93m",   # yellow
        "ERROR": "\033[91m",     # red
        "CRITICAL": "\033[1;91m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        line = super().format(record)
        return f"{color}{line}{self.RESET}"


def get_logger(name, level=logging.INFO):
    """
    Return a logger that prints colored messages to the terminal.

    INFO and above are shown by default. Safe to call more than once
    (e.g. re-running a notebook cell) without printing duplicate lines.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)  # stdout so notebooks don't show it with a red background
        handler.setFormatter(ColoredFormatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False  # avoid printing twice if the root logger is also configured
    return logger
