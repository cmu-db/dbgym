"""
"Infra" refers to benchmark/ and dbms/. These are all the paths used to access the files created by benchmark/ and dbms/.
They're inside gymlib because agents will need to access them.
"""

from pathlib import Path
from typing import Any

from gymlib.workspace import DBGYM_APP_NAME, SYMLINKS_DNAME, name_to_linkname

SCALE_FACTOR_PLACEHOLDER: str = "[scale_factor]"
BENCHMARK_NAME_PLACEHOLDER: str = "[benchmark_name]"
WORKLOAD_NAME_PLACEHOLDER: str = "[workload_name]"
DEFAULT_SCALE_FACTOR = 1.0


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


def get_dbdata_tgz_filename(benchmark_name: str, scale_factor: float | str) -> str:
    return f"{benchmark_name}_sf{get_scale_factor_string(scale_factor)}_pristine_dbdata.tgz"


def get_tables_symlink_path(
    workspace_path: Path, benchmark: str, scale_factor: float | str
) -> Path:
    return (
        workspace_path
        / SYMLINKS_DNAME
        / DBGYM_APP_NAME
        / name_to_linkname(get_tables_dirname(benchmark, scale_factor))
    )


def get_workload_symlink_path(
    workspace_path: Path, benchmark: str, scale_factor: float | str, suffix: str
) -> Path:
    return (
        workspace_path
        / SYMLINKS_DNAME
        / DBGYM_APP_NAME
        / name_to_linkname(get_workload_dirname(benchmark, scale_factor, suffix))
    )


def get_repo_symlink_path(workspace_path: Path) -> Path:
    return workspace_path / SYMLINKS_DNAME / DBGYM_APP_NAME / "repo.link"


def get_pgbin_symlink_path(workspace_path: Path) -> Path:
    return get_repo_symlink_path(workspace_path) / "boot" / "build" / "postgres" / "bin"


def get_dbdata_tgz_symlink_path(
    workspace_path: Path, benchmark_name: str, scale_factor: float | str
) -> Path:
    return (
        workspace_path
        / SYMLINKS_DNAME
        / DBGYM_APP_NAME
        / name_to_linkname(get_dbdata_tgz_filename(benchmark_name, scale_factor))
    )
