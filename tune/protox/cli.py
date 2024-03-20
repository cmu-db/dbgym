import click

from misc.utils import DBGymConfig
from tune.protox.agent.cli import agent_group
from tune.protox.embedding.cli import embedding_group


@click.group(name="protox")
@click.pass_obj
def protox_group(dbgym_cfg: DBGymConfig):
    dbgym_cfg.append_group("protox")

protox_group.add_command(embedding_group)
protox_group.add_command(agent_group)
