import logging
import os
import warnings
from logging import Logger
from pathlib import Path
from typing import Any, Optional

import click

# Do this to suppress the logs we'd usually get when importing tensorflow.
# By importing tensorflow in task.py, we avoid it being imported in any other file since task.py is always entered first.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow

del os.environ["TF_CPP_MIN_LOG_LEVEL"]

from benchmark.cli import benchmark_group
from dbms.cli import dbms_group
from manage.cli import manage_group
from misc.utils import DBGymConfig
from tune.cli import tune_group

# TODO(phw2): Save commit, git diff, and run command.
# TODO(phw2): Remove write permissions on old run_*/ dirs to enforce that they are immutable.
# TODO(phw2): Rename run_*/ to the command used (e.g. tune_protox_*/).


@click.group()
@click.pass_context
def task(ctx: click.Context) -> None:
    """🛢️ CMU-DB Database Gym: github.com/cmu-db/dbgym 🏋️"""
    dbgym_config_path = Path(os.getenv("DBGYM_CONFIG_PATH", "dbgym_config.yaml"))
    dbgym_cfg = DBGymConfig(dbgym_config_path)
    ctx.obj = dbgym_cfg

    _set_up_loggers(dbgym_cfg)
    _set_up_warnings(dbgym_cfg)


class LongLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if record.args:
            record.msg = str(record.msg) % record.args
        return super().format(record)


def _set_up_loggers(dbgym_cfg: DBGymConfig) -> None:
    """
    Set up everything related to the logging library.

    If your script needs to provide output, use the output logger (I usually use the info level). If you want to print things for
    debugging purposes, use print(). If you want to log things, use the logging library. When using the logging library, use the
    root logger unless you have a good reason not to.
    """
    # The output logger behaves identically to print. We use it to indicate that something is not a debugging print but rather
    # is actual output of the program.
    output_format = "%(message)s"
    _set_up_logger(
        logging.getLogger("output"),
        output_format,
        dbgym_cfg.cur_task_runs_artifacts_path(mkdir=True) / "output.log",
        console_level=logging.INFO,
    )

    # The root logger is set up globally here. Do not reconfigure the root logger anywhere else.
    log_format = "%(levelname)s:%(asctime)s [%(filename)s:%(lineno)s]  %(message)s"
    _set_up_logger(
        logging.getLogger(),
        log_format,
        dbgym_cfg.cur_task_runs_artifacts_path(mkdir=True) / "root.log",
    )

    # Set up some of the third-party loggers.
    # Make sure to clear the handlers to remove the console handler that tensorflow creates by default.
    for logger_name in ["tensorflow"]:
        logger = logging.root.manager.loggerDict[logger_name]
        assert isinstance(logger, Logger)
        logger.handlers.clear()
        _set_up_logger(
            logger,
            log_format,
            dbgym_cfg.cur_task_runs_artifacts_path(mkdir=True) / f"{logger_name}.log",
        )


def _set_up_logger(
    logger: Logger,
    format: str,
    output_log_fpath: Path,
    console_level: int = logging.ERROR,
    file_level: int = logging.DEBUG,
) -> None:
    # Set this so that the root logger captures everything.
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(format)

    # Only make it output warnings or higher to the console.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Let it output everything to the output file.
    file_handler = logging.FileHandler(output_log_fpath)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(file_level)
    logger.addHandler(file_handler)


def _set_up_warnings(dbgym_cfg: DBGymConfig) -> None:
    """
    Some libraries (like torch) use warnings instead of logging for warnings. I want to redirect these too to avoid cluttering the console.
    """
    warnings_fpath = dbgym_cfg.cur_task_runs_artifacts_path(mkdir=True) / "warnings.log"

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


if __name__ == "__main__":
    task.add_command(benchmark_group)
    task.add_command(manage_group)
    task.add_command(dbms_group)
    task.add_command(tune_group)
    task()
