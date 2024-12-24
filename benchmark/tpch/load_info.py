from pathlib import Path
from typing import Optional

from dbms.load_info_base_class import LoadInfoBaseClass
from util.workspace import DBGymConfig, get_default_tables_dname, is_fully_resolved

TPCH_SCHEMA_FNAME = "tpch_schema.sql"
TPCH_CONSTRAINTS_FNAME = "tpch_constraints.sql"


class TpchLoadInfo(LoadInfoBaseClass):
    # currently, hardcoding the path seems like the easiest solution. If the path ever changes, it'll
    # just break an integration test and we can fix it. I don't want to prematurely overengineer it
    CODEBASE_PATH_COMPONENTS = ["dbgym", "benchmark", "tpch"]
    CODEBASE_DNAME = "_".join(CODEBASE_PATH_COMPONENTS)
    TABLES = [
        "region",
        "nation",
        "part",
        "supplier",
        "partsupp",
        "customer",
        "orders",
        "lineitem",
    ]

    def __init__(self, dbgym_cfg: DBGymConfig, scale_factor: float):
        # schema and constraints
        schema_root_dpath = dbgym_cfg.dbgym_repo_path
        for component in TpchLoadInfo.CODEBASE_PATH_COMPONENTS[
            1:
        ]:  # [1:] to skip "dbgym"
            schema_root_dpath /= component
        self._schema_fpath = schema_root_dpath / TPCH_SCHEMA_FNAME
        assert (
            self._schema_fpath.exists()
        ), f"self._schema_fpath ({self._schema_fpath}) does not exist"
        self._constraints_fpath = schema_root_dpath / TPCH_CONSTRAINTS_FNAME
        assert (
            self._constraints_fpath.exists()
        ), f"self._constraints_fpath ({self._constraints_fpath}) does not exist"

        # tables
        data_root_dpath = (
            dbgym_cfg.dbgym_symlinks_path / TpchLoadInfo.CODEBASE_DNAME / "data"
        )
        tables_symlink_dpath = (
            data_root_dpath / f"{get_default_tables_dname(scale_factor)}.link"
        )
        tables_dpath = tables_symlink_dpath.resolve()
        assert is_fully_resolved(
            tables_dpath
        ), f"tables_dpath ({tables_dpath}) should be an existent real absolute path. Make sure you have generated the TPC-H data"
        self._tables_and_fpaths = []
        for table in TpchLoadInfo.TABLES:
            table_fpath = tables_dpath / f"{table}.tbl"
            self._tables_and_fpaths.append((table, table_fpath))

    def get_schema_fpath(self) -> Path:
        return self._schema_fpath

    def get_tables_and_fpaths(self) -> list[tuple[str, Path]]:
        return self._tables_and_fpaths

    def get_table_file_delimiter(self) -> str:
        return "|"

    def get_constraints_fpath(self) -> Optional[Path]:
        return self._constraints_fpath
