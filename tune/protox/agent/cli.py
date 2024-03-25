import click

from misc.utils import DBGymConfig
from tune.protox.agent.hpo import hpo


@click.group("agent")
@click.pass_obj
def agent_group(dbgym_cfg: DBGymConfig):
    dbgym_cfg.append_group("agent")


agent_group.add_command(hpo)
