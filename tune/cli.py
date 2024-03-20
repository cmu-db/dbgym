import click

from misc.utils import DBGymConfig
from tune.protox.cli import protox_group


@click.group(name="tune")
@click.pass_obj
def tune_group(dbgym_cfg: DBGymConfig):
    dbgym_cfg.append_group("tune")

tune_group.add_command(protox_group)
