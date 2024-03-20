import click

from dbms.postgres.cli import postgres_group


@click.group(name="dbms")
@click.pass_obj
def dbms_group(dbgym_cfg):
    dbgym_cfg.append_group("dbms")

dbms_group.add_command(postgres_group)
