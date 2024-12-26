import click

from util.workspace import DBGymConfig


@click.group(name="tune")
@click.pass_obj
def tune_group(dbgym_cfg: DBGymConfig) -> None:
    dbgym_cfg.append_group("tune")
