"""Structured Logging — Consistent logging across the application."""
import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # File handler
        from src.config import settings
        log_dir = settings.BASE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(level)
    return logger


def log_pipeline_step(logger: logging.Logger, step: str, details: str = ""):
    """Log a pipeline step with consistent formatting."""
    msg = f"STEP: {step}"
    if details:
        msg += f" — {details}"
    logger.info(msg)
