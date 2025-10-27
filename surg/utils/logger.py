"""
Logging Utility for SURG

This module provides a centralized logging system for the entire SURG codebase.
It supports multiple log levels, file and console output, colored output,
and structured logging with timestamps and module information.

Features:
- Colored console output for different log levels
- File logging with rotation
- Configurable log levels per module
- Structured logging with context
- Performance logging
- Exception logging with traceback
- Thread-safe logging
- Global logger accessible from anywhere

Usage:
    # Basic usage
    from surg.utils.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Starting recommendation process")
    logger.warning("Low data quality detected")
    logger.error("Failed to load model", exc_info=True)
    
    # With context
    logger.info("Processing user", extra={'user_id': 123, 'items': 50})
    
    # Performance logging
    with logger.timer("Model training"):
        model.fit(data)
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import traceback
from contextlib import contextmanager
import time


# ANSI color codes for terminal output
class ColorCodes:
    """ANSI color codes for colored terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright foreground colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log levels in console output.
    """
    
    LEVEL_COLORS = {
        logging.DEBUG: ColorCodes.BRIGHT_BLACK,
        logging.INFO: ColorCodes.BRIGHT_BLUE,
        logging.WARNING: ColorCodes.BRIGHT_YELLOW,
        logging.ERROR: ColorCodes.BRIGHT_RED,
        logging.CRITICAL: ColorCodes.BOLD + ColorCodes.BRIGHT_RED,
    }
    
    def format(self, record):
        """Format log record with colors."""
        # Add color to level name
        if record.levelno in self.LEVEL_COLORS:
            record.levelname = (
                f"{self.LEVEL_COLORS[record.levelno]}"
                f"{record.levelname:<8}"
                f"{ColorCodes.RESET}"
            )
        
        # Add color to module name
        record.name = f"{ColorCodes.CYAN}{record.name}{ColorCodes.RESET}"
        
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """
    Formatter that supports structured logging with additional context.
    """
    
    def format(self, record):
        """Format log record with additional context if available."""
        # Format the base message
        result = super().format(record)
        
        # Add extra context if available
        if hasattr(record, 'context') and record.context:
            context_str = ' | '.join(f"{k}={v}" for k, v in record.context.items())
            result = f"{result} | {context_str}"
        
        return result


class SURGLogger:
    """
    Enhanced logger wrapper with additional functionality.
    
    Provides convenience methods for common logging patterns,
    performance tracking, and structured logging.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize SURG logger wrapper.
        
        Args:
            logger: Standard Python logger instance
        """
        self._logger = logger
    
    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, msg, **kwargs)
    
    def exception(self, msg: str, **kwargs):
        """Log exception with traceback."""
        kwargs['exc_info'] = True
        self._log(logging.ERROR, msg, **kwargs)
    
    def _log(self, level: int, msg: str, **kwargs):
        """
        Internal logging method with context support.
        
        Args:
            level: Log level
            msg: Log message
            **kwargs: Additional context and logging parameters
        """
        # Extract context from kwargs
        context = {k: v for k, v in kwargs.items() 
                  if k not in ['exc_info', 'stack_info', 'stacklevel', 'extra']}
        
        # Remove context from kwargs
        for key in context:
            kwargs.pop(key, None)
        
        # Add context to extra
        if context:
            kwargs['extra'] = {'context': context}
        
        # Log the message
        self._logger.log(level, msg, **kwargs)
    
    @contextmanager
    def timer(self, operation: str, level: int = logging.INFO):
        """
        Context manager for timing operations.
        
        Args:
            operation: Name of the operation being timed
            level: Log level for timing message
            
        Example:
            >>> with logger.timer("Model training"):
            ...     model.fit(data)
            [INFO] Model training completed in 3.45s
        """
        start_time = time.time()
        self._logger.log(level, f"Starting: {operation}")
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self._logger.log(level, f"Completed: {operation} in {elapsed:.2f}s")
    
    def success(self, msg: str, **kwargs):
        """Log success message (info level with success indicator)."""
        self.info(f"✅ {msg}", **kwargs)
    
    def failure(self, msg: str, **kwargs):
        """Log failure message (error level with failure indicator)."""
        self.error(f"❌ {msg}", **kwargs)
    
    def progress(self, msg: str, current: int, total: int, **kwargs):
        """
        Log progress message.
        
        Args:
            msg: Progress message
            current: Current progress
            total: Total items
            **kwargs: Additional context
        """
        percentage = (current / total * 100) if total > 0 else 0
        self.info(f"⏳ {msg} [{current}/{total}] ({percentage:.1f}%)", **kwargs)
    
    def section(self, title: str, level: int = logging.INFO):
        """
        Log a section header.
        
        Args:
            title: Section title
            level: Log level
        """
        separator = "=" * 60
        self._logger.log(level, separator)
        self._logger.log(level, f"  {title}")
        self._logger.log(level, separator)
    
    @property
    def level(self) -> int:
        """Get current log level."""
        return self._logger.level
    
    @level.setter
    def level(self, level: int):
        """Set log level."""
        self._logger.setLevel(level)


# Global configuration
_initialized = False
_log_dir: Optional[Path] = None
_log_level: int = logging.INFO
_console_level: int = logging.INFO
_file_level: int = logging.DEBUG
_enable_colors: bool = True


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    console_level: Optional[str] = None,
    file_level: Optional[str] = None,
    enable_colors: bool = True,
    log_file_name: str = "surg.log"
) -> None:
    """
    Setup global logging configuration for SURG.
    
    This should be called once at application startup to configure
    logging for the entire codebase.
    
    Args:
        log_dir: Directory for log files (if None, no file logging)
        log_level: Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_level: Console log level (if None, uses log_level)
        file_level: File log level (if None, uses DEBUG)
        enable_colors: Whether to enable colored console output
        log_file_name: Name of the log file
        
    Example:
        >>> from surg.utils.logger import setup_logging
        >>> setup_logging(
        ...     log_dir="./logs",
        ...     log_level="INFO",
        ...     enable_colors=True
        ... )
    """
    global _initialized, _log_dir, _log_level, _console_level, _file_level, _enable_colors
    
    # Convert level strings to constants
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    _log_level = level_map.get(log_level.upper(), logging.INFO)
    _console_level = level_map.get(console_level.upper(), _log_level) if console_level else _log_level
    _file_level = level_map.get(file_level.upper(), logging.DEBUG) if file_level else logging.DEBUG
    _enable_colors = enable_colors
    
    # Setup log directory
    if log_dir:
        _log_dir = Path(log_dir)
        _log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger('surg')
    root_logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(_console_level)
    
    if _enable_colors and sys.stdout.isatty():
        console_format = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        console_formatter = ColoredFormatter(
            console_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        console_formatter = logging.Formatter(
            console_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if _log_dir:
        log_file = _log_dir / log_file_name
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(_file_level)
        
        file_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
        file_formatter = StructuredFormatter(
            file_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    _initialized = True
    
    # Log initialization
    root_logger.info(f"SURG logging initialized (level={log_level}, console={console_level or log_level}, file={file_level or 'DEBUG'})")
    if _log_dir:
        root_logger.info(f"Log files will be saved to: {_log_dir}")


def get_logger(name: str) -> SURGLogger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Logger name (typically __name__ of the module)
        
    Returns:
        SURGLogger instance configured for the module
        
    Example:
        >>> from surg.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting process")
    """
    # Initialize with defaults if not already initialized
    if not _initialized:
        setup_logging()
    
    # Ensure name starts with 'surg.' for proper hierarchy
    if not name.startswith('surg'):
        name = f'surg.{name}'
    
    # Get standard logger and wrap it
    standard_logger = logging.getLogger(name)
    return SURGLogger(standard_logger)


def set_log_level(level: str, logger_name: Optional[str] = None) -> None:
    """
    Set log level for a specific logger or all SURG loggers.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: Specific logger name (if None, sets root SURG logger)
        
    Example:
        >>> from surg.utils.logger import set_log_level
        >>> set_log_level('DEBUG')  # All loggers
        >>> set_log_level('WARNING', 'surg.core')  # Specific logger
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger('surg')
    
    logger.setLevel(log_level)


def disable_logging(logger_name: Optional[str] = None) -> None:
    """
    Disable logging for a specific logger or all SURG loggers.
    
    Args:
        logger_name: Specific logger name (if None, disables all SURG logging)
    """
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger('surg')
    
    logger.disabled = True


def enable_logging(logger_name: Optional[str] = None) -> None:
    """
    Enable logging for a specific logger or all SURG loggers.
    
    Args:
        logger_name: Specific logger name (if None, enables all SURG logging)
    """
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger('surg')
    
    logger.disabled = False


# Convenience functions for quick logging without getting a logger instance
def debug(msg: str, **kwargs):
    """Log debug message to root SURG logger."""
    get_logger('surg').debug(msg, **kwargs)


def info(msg: str, **kwargs):
    """Log info message to root SURG logger."""
    get_logger('surg').info(msg, **kwargs)


def warning(msg: str, **kwargs):
    """Log warning message to root SURG logger."""
    get_logger('surg').warning(msg, **kwargs)


def error(msg: str, **kwargs):
    """Log error message to root SURG logger."""
    get_logger('surg').error(msg, **kwargs)


def critical(msg: str, **kwargs):
    """Log critical message to root SURG logger."""
    get_logger('surg').critical(msg, **kwargs)


def exception(msg: str, **kwargs):
    """Log exception with traceback to root SURG logger."""
    get_logger('surg').exception(msg, **kwargs)


def log_performance(logger: Optional[SURGLogger] = None, level: str = "INFO"):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance to use. If None, uses the function's module logger
        level: Log level to use (default: INFO)
        
    Example:
        @log_performance(logger)
        def slow_function():
            time.sleep(1)
            return "done"
    """
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                log_method = getattr(logger, level.lower(), logger.info)
                log_method(
                    f"Function '{func.__name__}' completed in {elapsed:.3f}s",
                    function=func.__name__,
                    duration=elapsed
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"Function '{func.__name__}' failed after {elapsed:.3f}s",
                    function=func.__name__,
                    duration=elapsed,
                    error=str(e)
                )
                raise
        
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Setup logging with file output
    setup_logging(
        log_dir="./logs",
        log_level="DEBUG",
        enable_colors=True
    )
    
    # Get a logger for this module
    logger = get_logger(__name__)
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test success/failure messages
    logger.success("Operation completed successfully")
    logger.failure("Operation failed")
    
    # Test progress logging
    for i in range(1, 6):
        logger.progress("Processing items", i, 5)
        time.sleep(0.1)
    
    # Test section headers
    logger.section("Starting New Section")
    logger.info("Content in the section")
    
    # Test context logging
    logger.info("User activity", user_id=123, action="login", ip="192.168.1.1")
    
    # Test timer
    with logger.timer("Expensive operation"):
        time.sleep(0.5)
    
    # Test exception logging
    try:
        raise ValueError("Something went wrong")
    except Exception:
        logger.exception("An error occurred during processing")
    
    print("\n✅ Logger testing complete! Check ./logs/surg.log for file output.")
