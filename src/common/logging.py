"""Logging setup — call configure_logging() once at process startup.

Uses loguru, configured from the LoggingConfig section of settings.
Console output is colorized for readability; file output is plain text.
"""
import sys

from loguru import logger

from src.common.paths import LOGS_DIR


def configure_logging(
    level: str = "INFO",
    to_file: bool = True,
    rotation: str = "00:00",
    retention: str = "14 days",
) -> None:
    """Configure loguru sinks. Idempotent — safe to call multiple times."""
    logger.remove()  # clear default handler

    # Console (colorized).
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <7}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    # File (plain text, daily rotation).
    if to_file:
        logger.add(
            LOGS_DIR / "rag_{time:YYYY-MM-DD}.log",
            level=level,
            rotation=rotation,
            retention=retention,
            enqueue=True,  # thread-safe writes
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | "
                "{name}:{function}:{line} - {message}"
            ),
        )
