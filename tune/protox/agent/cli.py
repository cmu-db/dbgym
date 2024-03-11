import click

from misc.utils import DBGymConfig


@click.group("agent")
@click.pass_obj
def agent_group(config: DBGymConfig):
    config.append_group("agent")
