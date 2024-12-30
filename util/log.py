import logging
import warnings
from logging import Logger
from pathlib import Path
from typing import Any, Optional

DBGYM_LOGGER_NAME = "dbgym"
DBGYM_OUTPUT_LOGGER_NAME = f"{DBGYM_LOGGER_NAME}.output"


def set_up_loggers(log_path: Path) -> None:
    """
    Set up everything related to the logging library.

    If your script needs to provide output, use the output logger (I usually use the info level). If you want to print things for
    debugging purposes, use print(). If you want to log things, use the dbgym logger.
    """

    # The dbgym logger is set up globally here. Do not reconfigure the dbgym logger anywhere else.
    log_format = "%(levelname)s:%(asctime)s [%(filename)s:%(lineno)s]  %(message)s"
    _set_up_logger(
        logging.getLogger(DBGYM_LOGGER_NAME),
        log_format,
        log_path / f"{DBGYM_LOGGER_NAME}.log",
    )

    # The output logger is meant to output things to the console. We use it instead of using print to indicate that something is
    # not a debugging print but rather is actual output of the program.
    # We pass it None so that it doesn't write to its own file. However, due to the logging hierarchy, all messages logged to
    # the output logger will be propagated to the dbgym logger and will thus be written to its file.
    output_format = "%(message)s"
    _set_up_logger(
        logging.getLogger(DBGYM_OUTPUT_LOGGER_NAME),
        output_format,
        None,
        console_level=logging.DEBUG,
    )


def _set_up_logger(
    logger: Logger,
    format: str,
    output_log_path: Optional[Path],
    console_level: int = logging.ERROR,
    file_level: int = logging.DEBUG,
) -> None:
    # Set this so that the logger captures everything.
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(format)

    # Only make it output warnings or higher to the console.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Let it output everything to the output file.
    if output_log_path is not None:
        file_handler = logging.FileHandler(output_log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level)
        logger.addHandler(file_handler)


def set_up_warnings(log_path: Path) -> None:
    """
    Some libraries (like torch) use warnings instead of logging for warnings. I want to redirect these too to avoid cluttering the console.
    """
    warnings_path = log_path / "warnings.log"

    def write_warning_to_file(
        message: Any,
        category: Any,
        filename: Any,
        lineno: Any,
        file: Optional[Any] = None,
        line: Optional[Any] = None,
    ) -> None:
        with open(warnings_path, "a") as f:
            f.write(f"{filename}:{lineno}: {category.__name__}: {message}\n")

    warnings.showwarning = write_warning_to_file
