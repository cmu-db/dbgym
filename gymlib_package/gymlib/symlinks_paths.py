from pathlib import Path
from typing import Any

# TODO: move these into workspace.py and move workspace.py into gymlib.
SYMLINKS_DNAME = "symlinks"
DBGYM_APP_NAME = "dbgym"

SCALE_FACTOR_PLACEHOLDER: str = "[scale_factor]"
BENCHMARK_NAME_PLACEHOLDER: str = "[benchmark_name]"
WORKLOAD_NAME_PLACEHOLDER: str = "[workload_name]"


def get_scale_factor_string(scale_factor: float | str) -> str:
    if type(scale_factor) is str and scale_factor == SCALE_FACTOR_PLACEHOLDER:
        return scale_factor
    else:
        if float(int(scale_factor)) == scale_factor:
            return str(int(scale_factor))
        else:
            return str(scale_factor).replace(".", "point")


def get_tables_dirname(benchmark: str, scale_factor: float | str) -> str:
    return f"tables_{benchmark}_sf{get_scale_factor_string(scale_factor)}"


def get_workload_suffix(benchmark: str, **kwargs: Any) -> str:
    if benchmark == "tpch":
        assert kwargs.keys() == {"seed_start", "seed_end", "query_subset"}
        return f"{kwargs['seed_start']}_{kwargs['seed_end']}_{kwargs['query_subset']}"
    elif benchmark == "job":
        assert kwargs.keys() == {"query_subset"}
        return f"{kwargs['query_subset']}"
    else:
        assert False


def get_workload_dirname(benchmark: str, scale_factor: float | str, suffix: str) -> str:
    return f"workload_{benchmark}_sf{get_scale_factor_string(scale_factor)}_{suffix}"


def get_tables_symlink_path(
    workspace_path: Path, benchmark: str, scale_factor: float | str
) -> Path:
    return (
        workspace_path
        / SYMLINKS_DNAME
        / DBGYM_APP_NAME
        / (get_tables_dirname(benchmark, scale_factor) + ".link")
    )


def get_workload_symlink_path(
    workspace_path: Path, benchmark: str, scale_factor: float | str, suffix: str
) -> Path:
    return (
        workspace_path
        / SYMLINKS_DNAME
        / DBGYM_APP_NAME
        / (get_workload_dirname(benchmark, scale_factor, suffix) + ".link")
    )
