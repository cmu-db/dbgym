import click

from dbms.postgres.cli import postgres_group
from util.workspace import DBGymWorkspace


@click.group(name="dbms")
@click.pass_obj
def dbms_group(dbgym_workspace: DBGymWorkspace) -> None:
    pass


dbms_group.add_command(postgres_group)
