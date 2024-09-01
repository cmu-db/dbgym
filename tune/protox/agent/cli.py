import click

from misc.utils import DBGymConfig
from tune.protox.agent.hpo import hpo
from tune.protox.agent.replay import replay
from tune.protox.agent.tune import tune


@click.group("agent")
@click.pass_obj
def agent_group(dbgym_cfg: DBGymConfig) -> None:
    dbgym_cfg.append_group("agent")


agent_group.add_command(hpo)
agent_group.add_command(tune)
agent_group.add_command(replay)
