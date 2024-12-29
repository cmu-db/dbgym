from pathlib import Path
from typing import Optional

from gymlib.symlinks_paths import get_tables_symlink_path

from benchmark.constants import DEFAULT_SCALE_FACTOR
from dbms.load_info_base_class import LoadInfoBaseClass
from util.workspace import DBGymWorkspace, fully_resolve_path

JOB_SCHEMA_FNAME = "job_schema.sql"


class JobLoadInfo(LoadInfoBaseClass):
    TABLES = [
        "aka_name",
        "aka_title",
        "cast_info",
        "char_name",
        "comp_cast_type",
        "company_name",
        "company_type",
        "complete_cast",
        "info_type",
        "keyword",
        "kind_type",
        "link_type",
        "movie_companies",
        "movie_info",
        "movie_info_idx",
        "movie_keyword",
        "movie_link",
        "name",
        "person_info",
        "role_type",
        "title",
    ]

    def __init__(self, dbgym_workspace: DBGymWorkspace):
        # Schema (directly in the codebase).
        job_codebase_path = dbgym_workspace.base_dbgym_repo_dpath / "benchmark" / "job"
        self._schema_fpath = job_codebase_path / JOB_SCHEMA_FNAME
        assert (
            self._schema_fpath.exists()
        ), f"self._schema_fpath ({self._schema_fpath}) does not exist"

        # Tables
        tables_path = fully_resolve_path(
            get_tables_symlink_path(
                dbgym_workspace.dbgym_workspace_path, "job", DEFAULT_SCALE_FACTOR
            )
        )
        self._tables_and_paths = []
        for table in JobLoadInfo.TABLES:
            table_fpath = tables_path / f"{table}.csv"
            self._tables_and_paths.append((table, table_fpath))

    def get_schema_fpath(self) -> Path:
        return self._schema_fpath

    def get_tables_and_paths(self) -> list[tuple[str, Path]]:
        return self._tables_and_paths

    def get_table_file_delimiter(self) -> str:
        return ","

    def get_constraints_fpath(self) -> Optional[Path]:
        # JOB does not have any constraints. It does have indexes, but we don't want to create
        # those indexes so that the tuning agent can start from a clean slate.
        return None
