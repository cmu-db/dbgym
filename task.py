import click
from gymlib.workspace import make_standard_dbgym_workspace

from benchmark.cli import benchmark_group
from dbms.cli import dbms_group
from manage.cli import manage_group

# TODO(phw2): Save commit, git diff, and run command.
# TODO(phw2): Remove write permissions on old run_*/ dirs to enforce that they are immutable.
# TODO(phw2): Rename run_*/ to the command used (e.g. tune_protox_*/).


@click.group()
@click.pass_context
def task(ctx: click.Context) -> None:
    """🛢️ CMU-DB Database Gym: github.com/cmu-db/dbgym 🏋️"""
    dbgym_workspace = make_standard_dbgym_workspace()
    ctx.obj = dbgym_workspace


if __name__ == "__main__":
    task.add_command(benchmark_group)
    task.add_command(manage_group)
    task.add_command(dbms_group)
    task()
