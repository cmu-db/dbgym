import streamlit as st

from tune.env.pg_conn import PostgresConn
from util.pg import DEFAULT_POSTGRES_PORT
from util.workspace import (
    DBGymConfig,
    make_standard_dbgym_cfg,
    DEFAULT_BOOT_CONFIG_FPATH,
    default_dbdata_parent_dpath,
    default_pgbin_path,
    default_pristine_dbdata_snapshot_path,
)


# This ensures that DBGymConfig is only created once. Check DBGymConfig.__init__() for why we must do this.
@st.cache_resource
def make_dbgym_cfg() -> DBGymConfig:
    return make_standard_dbgym_cfg()


class Demo:
    BENCHMARK = "tpch"
    SCALE_FACTOR = 0.01

    def __init__(self):
        self.dbgym_cfg = make_dbgym_cfg()
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
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = True
        demo = Demo()

    st.write("hia")