import streamlit as st

from tune.env.pg_conn import PostgresConn
from util.pg import DEFAULT_POSTGRES_PORT
from util.workspace import (
    make_standard_dbgym_cfg,
    DEFAULT_BOOT_CONFIG_FPATH,
    default_dbdata_parent_dpath,
    default_pgbin_path,
    default_pristine_dbdata_snapshot_path,
)


class Demo:
    BENCHMARK = "tpch"
    SCALE_FACTOR = 0.01

    def __init__(self):
        self.dbgym_cfg = make_standard_dbgym_cfg()
        self.pristine_dbdata_snapshot_path = default_pristine_dbdata_snapshot_path(
            self.dbgym_cfg.dbgym_workspace_path, Demo.BENCHMARK, Demo.SCALE_FACTOR
        )
        self.dbdata_parent_dpath = default_dbdata_parent_dpath(
            self.dbgym_cfg.dbgym_workspace_path
        )
        self.pgbin_dpath = default_pgbin_path(self.dbgym_cfg.dbgym_workspace_path)
        self.pg_conn = PostgresConn(
            self.dbgym_cfg,
            DEFAULT_POSTGRES_PORT,
            self.pristine_dbdata_snapshot_path,
            self.dbdata_parent_dpath,
            self.pgbin_dpath,
            False,
            DEFAULT_BOOT_CONFIG_FPATH,
        )


if __name__ == "__main__":
    demo = Demo()