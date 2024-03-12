import click

from misc.utils import DBGymConfig


@click.group("agent")
@click.pass_obj
def agent_group(cfg: DBGymConfig):
    cfg.append_group("agent")
