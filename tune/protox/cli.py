import click

from tune.protox.agent.cli import agent_group
from tune.protox.embedding.cli import embedding_group
from util.workspace import DBGymConfig


@click.group(name="protox")
@click.pass_obj
def protox_group(dbgym_cfg: DBGymConfig) -> None:
    dbgym_cfg.append_group("protox")


protox_group.add_command(embedding_group)
protox_group.add_command(agent_group)
