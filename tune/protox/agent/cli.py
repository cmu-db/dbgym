import click

from misc.utils import DBGymConfig


@click.group("agent")
@click.pass_obj
def agent_group(dbgym_cfg: DBGymConfig):
    dbgym_cfg.append_group("agent")
