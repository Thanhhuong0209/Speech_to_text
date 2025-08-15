"""
Logging configuration for STT system.

This module provides centralized logging setup with configurable levels,
formats, and output destinations.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger

from .config import get_config


def setup_logger(name: str, level: Optional[str] = None) -> logger:
    """
    Setup logger for a module.
    
    Args:
        name: Module name
        level: Log level (optional, uses config default if None)
        
    Returns:
        Configured logger instance
    """
    config = get_config()
    log_level = level or config.log_level
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=log_level.upper(),
        colorize=True
    )
    
    # Add file handler if configured
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            config.log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level.upper(),
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
    
    # Intercept standard logging
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # Configure standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    return logger.bind(name=name)


def get_logger(name: str) -> logger:
    """
    Get logger for a module (alias for setup_logger).
    
    Args:
        name: Module name
        
    Returns:
        Logger instance
    """
    return setup_logger(name)


# Example usage
if __name__ == "__main__":
    # Test logging setup
    test_logger = setup_logger("test_module")
    
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    
    print("Logging setup test completed!")
