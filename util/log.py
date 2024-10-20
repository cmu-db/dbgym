import logging
from logging import Logger
from pathlib import Path
from typing import Any, Optional
import warnings


DBGYM_LOGGER_NAME = "dbgym"
DBGYM_OUTPUT_LOGGER_NAME = f"{DBGYM_LOGGER_NAME}.output"


def set_up_loggers(log_dpath: Path) -> None:
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
        log_dpath / f"{DBGYM_LOGGER_NAME}.log",
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

    # Set up some of the third-party loggers.
    # The reason I only set up a few select keys is to avoid cluttering the artifacts/ directory with too many *.log files.
    for logger_name in ["tensorflow", "ray"]:
        logger = logging.root.manager.loggerDict[logger_name]
        assert isinstance(logger, Logger)
        # Make sure to clear the handlers to remove the console handler that the loggers create by default.
        logger.handlers.clear()
        _set_up_logger(
            logger,
            log_format,
            log_dpath / f"{logger_name}.log",
        )


def _set_up_logger(
    logger: Logger,
    format: str,
    output_log_fpath: Optional[Path],
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
    if output_log_fpath is not None:
        file_handler = logging.FileHandler(output_log_fpath)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level)
        logger.addHandler(file_handler)


def set_up_warnings(log_dpath: Path) -> None:
    """
    Some libraries (like torch) use warnings instead of logging for warnings. I want to redirect these too to avoid cluttering the console.
    """
    warnings_fpath = log_dpath / "warnings.log"

    def write_warning_to_file(
        message: Any,
        category: Any,
        filename: Any,
        lineno: Any,
        file: Optional[Any] = None,
        line: Optional[Any] = None,
    ) -> None:
        with open(warnings_fpath, "a") as f:
            f.write(f"{filename}:{lineno}: {category.__name__}: {message}\n")

    warnings.showwarning = write_warning_to_file
