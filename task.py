import os
from pathlib import Path

import click

from util.log import set_up_loggers, set_up_warnings

# Do this to suppress the logs we'd usually get when importing tensorflow.
# By importing tensorflow in task.py, we avoid it being imported in any other file since task.py is always entered first.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow

del os.environ["TF_CPP_MIN_LOG_LEVEL"]

from benchmark.cli import benchmark_group
from dbms.cli import dbms_group
from manage.cli import manage_group
from analyze.cli import analyze_group
from tune.cli import tune_group
from util.workspace import DBGymConfig

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

    log_dpath = dbgym_cfg.cur_task_runs_artifacts_path(mkdir=True)
    set_up_loggers(log_dpath)
    set_up_warnings(log_dpath)


if __name__ == "__main__":
    task.add_command(benchmark_group)
    task.add_command(manage_group)
    task.add_command(analyze_group)
    task.add_command(dbms_group)
    task.add_command(tune_group)
    task()
