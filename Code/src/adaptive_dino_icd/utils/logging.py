"""
Logging utilities for training and evaluation.

Provides consistent logging setup across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def setup_logging(
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    name: str = "adaptive_dino_icd",
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_dir: Directory to save log files (if log_to_file=True)
        level: Logging level (default: INFO)
        name: Logger name
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_to_file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "adaptive_dino_icd") -> logging.Logger:
    """
    Get or create a logger with the given name.

    Args:
        name: Logger name (usually module name)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class MetricsLogger:
    """
    Logger for training metrics with JSON Lines output.

    Example:
        metrics_logger = MetricsLogger("./logs/metrics.jsonl")
        metrics_logger.log({"epoch": 1, "loss": 0.5, "accuracy": 0.9})
    """

    def __init__(self, log_path: Union[str, Path]):
        """
        Initialize metrics logger.

        Args:
            log_path: Path to JSON Lines file for metrics
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict) -> None:
        """
        Log metrics to file.

        Args:
            metrics: Dictionary of metric names and values
        """
        import json

        # Add timestamp
        metrics["timestamp"] = datetime.now().isoformat()

        with open(self.log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def read_all(self) -> list:
        """
        Read all logged metrics.

        Returns:
            List of metric dictionaries
        """
        import json

        if not self.log_path.exists():
            return []

        metrics = []
        with open(self.log_path, "r") as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
        return metrics
