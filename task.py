import logging
import os
from pathlib import Path

import click

from benchmark.cli import benchmark_group
from dbms.cli import dbms_group
from manage.cli import manage_group
from misc.utils import DBGymConfig
from tune.cli import tune_group

# TODO(phw2): save commit, git diff, and run command
# TODO(phw2): remove write permissions on old run_*/ dirs to enforce that they are immutable


@click.group()
@click.pass_context
def task(ctx: click.Context) -> None:
    """💩💩💩 CMU-DB Database Gym: github.com/cmu-db/dbgym 💩💩💩"""
    dbgym_config_path = Path(os.getenv("DBGYM_CONFIG_PATH", "dbgym_config.yaml"))
    dbgym_cfg = DBGymConfig(dbgym_config_path)
    ctx.obj = dbgym_cfg

    # The root logger is set up globally here. Do not reconfigure the root logger anywhere else.
    _set_up_root_logger(dbgym_cfg)


def _set_up_root_logger(dbgym_cfg: DBGymConfig) -> None:
    format = "%(levelname)s:%(asctime)s [%(filename)s:%(lineno)s]  %(message)s"
    
    # Set this so that the root logger captures everything.
    logging.getLogger().setLevel(logging.DEBUG)

    # Only make it output warnings or higher to the console.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(format))
    logging.getLogger().addHandler(console_handler)

    # Let it output everything to the output file.
    file_handler = logging.FileHandler(dbgym_cfg.cur_task_runs_artifacts_path(mkdir=True) / "output.log")
    file_handler.setFormatter(logging.Formatter(format))
    file_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(file_handler)


if __name__ == "__main__":
    task.add_command(benchmark_group)
    task.add_command(manage_group)
    task.add_command(dbms_group)
    task.add_command(tune_group)
    task()
