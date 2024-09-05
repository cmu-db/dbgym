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
    ctx.obj = DBGymConfig(dbgym_config_path)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s:%(name)s:%(levelname)s - %(message)s", level=logging.INFO
    )

    task.add_command(benchmark_group)
    task.add_command(manage_group)
    task.add_command(dbms_group)
    task.add_command(tune_group)
    task()
