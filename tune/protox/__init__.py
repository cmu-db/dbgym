from tune.protox.agent.cli import agent_group
from tune.protox.cli import protox_group
from tune.protox.embedding.cli import embedding_group

protox_group.add_command(embedding_group)
protox_group.add_command(agent_group)
