import streamlit as st

from env.pg_conn import PostgresConn
from util.pg import DEFAULT_POSTGRES_PORT, get_is_postgres_running
from util.workspace import (
    DEFAULT_BOOT_CONFIG_FPATH,
    DBGymConfig,
    default_dbdata_parent_dpath,
    default_pgbin_path,
    default_pristine_dbdata_snapshot_path,
    make_standard_dbgym_cfg,
)


# This ensures that DBGymConfig is only created once. Check DBGymConfig.__init__() for why we must do this.
@st.cache_resource
def make_dbgym_cfg() -> DBGymConfig:
    return make_standard_dbgym_cfg()


class Demo:
    BENCHMARK = "tpch"
    SCALE_FACTOR = 0.01

    def __init__(self) -> None:
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

    def get_categorized_system_knobs(self) -> tuple[dict[str, str], dict[str, str]]:
        IMPORTANT_KNOBS = {"shared_buffers", "wal_buffers"}
        all_knobs = self.pg_conn.get_system_knobs()
        important_knobs = {knob: val for knob, val in all_knobs.items() if knob in IMPORTANT_KNOBS}
        unimportant_knobs = {knob: val for knob, val in all_knobs.items() if knob not in IMPORTANT_KNOBS}
        return important_knobs, unimportant_knobs

    def main(self) -> None:
        is_postgres_running = get_is_postgres_running()

        if is_postgres_running:
            st.write("Postgres is running")

            if st.button("Stop Postgres"):
                self.pg_conn.shutdown_postgres()
                st.rerun()
            
            important_knobs, unimportant_knobs = self.get_categorized_system_knobs()
            with st.expander("Important knobs", expanded=True):
                st.json(important_knobs)

            with st.expander("Other knobs", expanded=False):
                st.json(unimportant_knobs)
        else:
            st.write("Postgres is not running")

            if st.button("Start Postgres"):
                self.pg_conn.restore_pristine_snapshot()
                self.pg_conn.restart_postgres()
                st.rerun()


if __name__ == "__main__":
    demo = Demo()
    demo.main()
