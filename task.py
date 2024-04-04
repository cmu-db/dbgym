import logging
import click

from misc.utils import DBGymConfig
from benchmark.cli import benchmark_group
from dbms.cli import dbms_group
from experiment.cli import experiment_group
from tune.cli import tune_group
from manage.cli import manage_group

# TODO(phw2): save commit, git diff, and run command
# TODO(phw2): remove write permissions on old run_*/ dirs to enforce that they are immutable


@click.group()
@click.option("--config-path", default="config.yaml")
@click.option("--no-startup-check", is_flag=True)
@click.pass_context
def task(ctx, config_path, no_startup_check):
    """💩💩💩 CMU-DB Database Gym: github.com/cmu-db/dbgym 💩💩💩"""
    ctx.obj = DBGymConfig(config_path, startup_check=not no_startup_check)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s:%(name)s:%(levelname)s - %(message)s", level=logging.INFO
    )

    task.add_command(benchmark_group)
    task.add_command(manage_group)
    task.add_command(dbms_group)
    task.add_command(experiment_group)
    task.add_command(tune_group)
    task()
