from tune.cli import tune_group
from tune.protox.cli import protox_group

tune_group.add_command(protox_group)
