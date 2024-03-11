from tune.protox.cli import protox_group
from tune.protox.embedding.cli import embedding_group
from tune.protox.agent.cli import agent_group

protox_group.add_command(agent_group)
