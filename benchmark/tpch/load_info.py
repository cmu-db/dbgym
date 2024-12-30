from pathlib import Path
from typing import Optional

from gymlib.symlinks_paths import get_tables_symlink_path

from dbms.load_info_base_class import LoadInfoBaseClass
from util.workspace import DBGymWorkspace, fully_resolve_path

TPCH_SCHEMA_FNAME = "tpch_schema.sql"
TPCH_CONSTRAINTS_FNAME = "tpch_constraints.sql"


class TpchLoadInfo(LoadInfoBaseClass):
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

    def __init__(self, dbgym_workspace: DBGymWorkspace, scale_factor: float):
        # Schema and constraints (directly in the codebase).
        tpch_codebase_path = dbgym_workspace.base_dbgym_repo_path / "benchmark" / "tpch"
        self._schema_path = tpch_codebase_path / TPCH_SCHEMA_FNAME
        assert (
            self._schema_path.exists()
        ), f"self._schema_path ({self._schema_path}) does not exist"
        self._constraints_path = tpch_codebase_path / TPCH_CONSTRAINTS_FNAME
        assert (
            self._constraints_path.exists()
        ), f"self._constraints_path ({self._constraints_path}) does not exist"

        # Tables
        tables_path = fully_resolve_path(
            get_tables_symlink_path(
                dbgym_workspace.dbgym_workspace_path, "tpch", scale_factor
            )
        )
        self._tables_and_paths = []
        for table in TpchLoadInfo.TABLES:
            table_path = tables_path / f"{table}.tbl"
            self._tables_and_paths.append((table, table_path))

    def get_schema_path(self) -> Path:
        return self._schema_path

    def get_tables_and_paths(self) -> list[tuple[str, Path]]:
        return self._tables_and_paths

    def get_table_file_delimiter(self) -> str:
        return "|"

    def get_constraints_path(self) -> Optional[Path]:
        return self._constraints_path
