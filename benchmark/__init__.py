from benchmark.cli import benchmark_group
from benchmark.tpch.cli import tpch_group

benchmark_group.add_command(tpch_group)
