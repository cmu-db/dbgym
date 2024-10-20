import logging
from logging import Logger
import os
from pathlib import Path
from typing import Any, Optional
import warnings

import click

# Do this to suppress the logs we'd usually get when importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
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


def _set_up_loggers(dbgym_cfg: DBGymConfig) -> None:
    """
    Set up everything related to the logging library.

    If you want to log things for real, use the logging library. Use the root logger unless you have a reason not to.

    If you want to print things for debugging purposes, use print(). Other than this, don"t use print().
    """
    format = "%(levelname)s:%(asctime)s [%(filename)s:%(lineno)s]  %(message)s"

    # The root logger is set up globally here. Do not reconfigure the root logger anywhere else.
    _set_up_logger(logging.getLogger(), format, dbgym_cfg.cur_task_runs_artifacts_path(mkdir=True) / "root.log")

    # Set up some of the third-party loggers.
    # Make sure to clear the handlers to remove the console handler that tensorflow creates by default.
    for logger_name in ["tensorflow"]:
        logger = logging.root.manager.loggerDict[logger_name]
        assert isinstance(logger, Logger)
        logger.handlers.clear()
        _set_up_logger(logger, format, dbgym_cfg.cur_task_runs_artifacts_path(mkdir=True) / f"{logger_name}.log")


def _set_up_logger(logger: Logger, format: str, output_log_fpath: Path, console_level: int=logging.ERROR, file_level: int=logging.DEBUG) -> None:
    # Set this so that the root logger captures everything.
    logger.setLevel(logging.DEBUG)

    # Only make it output warnings or higher to the console.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(console_handler)

    # Let it output everything to the output file.
    file_handler = logging.FileHandler(output_log_fpath)
    file_handler.setFormatter(logging.Formatter(format))
    file_handler.setLevel(file_level)
    logger.addHandler(file_handler)


def _set_up_warnings(dbgym_cfg: DBGymConfig) -> None:
    """
    Some libraries (like torch) use warnings instead of logging for warnings. I want to redirect these too to avoid cluttering the console.
    """
    warnings_fpath = dbgym_cfg.cur_task_runs_artifacts_path(mkdir=True) / "warnings.log"

    def write_warning_to_file(message: Any, category: Any, filename: Any, lineno: Any, file: Optional[Any]=None, line: Optional[Any]=None):
        with open(warnings_fpath, "a") as f:
            f.write(f"{filename}:{lineno}: {category.__name__}: {message}\n")

    warnings.showwarning = write_warning_to_file



if __name__ == "__main__":
    task.add_command(benchmark_group)
    task.add_command(manage_group)
    task.add_command(dbms_group)
    task.add_command(tune_group)
    task()
