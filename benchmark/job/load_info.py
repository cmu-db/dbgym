from pathlib import Path
from typing import Optional

from benchmark.constants import DEFAULT_SCALE_FACTOR
from dbms.load_info_base_class import LoadInfoBaseClass
from util.workspace import DBGymConfig, default_tables_dname, is_fully_resolved

JOB_SCHEMA_FNAME = "job_schema.sql"


class JobLoadInfo(LoadInfoBaseClass):
    CODEBASE_PATH_COMPONENTS = ["dbgym", "benchmark", "job"]
    CODEBASE_DNAME = "_".join(CODEBASE_PATH_COMPONENTS)
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

    def __init__(self, dbgym_cfg: DBGymConfig):
        # schema and constraints
        schema_root_dpath = dbgym_cfg.dbgym_repo_path
        for component in JobLoadInfo.CODEBASE_PATH_COMPONENTS[
            1:
        ]:  # [1:] to skip "dbgym"
            schema_root_dpath /= component
        self._schema_fpath = schema_root_dpath / JOB_SCHEMA_FNAME
        assert (
            self._schema_fpath.exists()
        ), f"self._schema_fpath ({self._schema_fpath}) does not exist"

        # Tables
        data_root_dpath = (
            dbgym_cfg.dbgym_symlinks_path / JobLoadInfo.CODEBASE_DNAME / "data"
        )
        tables_symlink_dpath = (
            data_root_dpath / f"{default_tables_dname(DEFAULT_SCALE_FACTOR)}.link"
        )
        tables_dpath = tables_symlink_dpath.resolve()
        assert is_fully_resolved(
            tables_dpath
        ), f"tables_dpath ({tables_dpath}) should be an existent real absolute path. Make sure you have generated the TPC-H data"
        self._tables_and_fpaths = []
        for table in JobLoadInfo.TABLES:
            table_fpath = tables_dpath / f"{table}.csv"
            self._tables_and_fpaths.append((table, table_fpath))

    def get_schema_fpath(self) -> Path:
        return self._schema_fpath

    def get_tables_and_fpaths(self) -> list[tuple[str, Path]]:
        return self._tables_and_fpaths

    def get_table_file_delimiter(self) -> str:
        return ","

    def get_constraints_fpath(self) -> Optional[Path]:
        # JOB does not have any constraints. It does have indexes, but we don't want to create
        # those indexes so that Proto-X can start from a clean slate.
        return None
