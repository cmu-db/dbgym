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


# The rationale behind this code is very subtle. I'll first go over streamlit concepts before describing why this function exists.
#
# First, in streamlit, there are three kinds of "script reruns". These are ordered from least to most "disruptive":
#  1. st.rerun(). Will reset any local variables but will not reset st.session_state.
#  2. Reloading the browser page (perhaps if you changed some code). Will reset local vars and st.session_state but not things
#     cached with @st.cache_resource.
#  3. Restarting the streamlit server. If you're running the server locally, you can restart it by doing Ctrl-C, `pkill python`,
#     and then `streamlit run ...` (or `./scripts/run_demo.sh`). Will reset local vars, st.session_state, and things cached with
#     @st.cache_resource, but will not reset things persisted to disk (though we currently don't persist anything to disk). Doing
#     `pkill python` is critical here to actually reset the things cached with @st.cache_resource.
#
# Next, DBGymConfig has a safeguard where it can only be created once per instance of the Python interpreter. If you just put it
# in st.session_state, it would get re-created when you reloaded the browser page, causing it to trip the assertion that checks
# DBGymConfig.num_times_created_this_run == 1. Thus, we use @st.cache_resource to avoid this.
#
# I considered modifying num_times_created_this_run to instead be num_active_instances and doing `num_active_instances -= 1` in
# DBGymConfig.__del__(). However, streamlit doesn't actually destroy objects when you reload the browser page; it only destroys
# objects when you restart the streamlit server.
#
# If you modify the code of DBGymConfig, you will need to fully restart the streamlit server for those changes to be propagated.
@st.cache_resource
def make_dbgym_cfg_cached() -> DBGymConfig:
    return make_standard_dbgym_cfg()


class Demo:
    BENCHMARK = "tpch"
    SCALE_FACTOR = 0.01

    def __init__(self) -> None:
        self.dbgym_cfg = make_dbgym_cfg_cached()
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

    def _get_categorized_system_knobs(self) -> tuple[dict[str, str], dict[str, str]]:
        IMPORTANT_KNOBS = {"shared_buffers", "enable_nestloop"}
        all_knobs = self.pg_conn.get_system_knobs()
        important_knobs = {
            knob: val for knob, val in all_knobs.items() if knob in IMPORTANT_KNOBS
        }
        unimportant_knobs = {
            knob: val for knob, val in all_knobs.items() if knob not in IMPORTANT_KNOBS
        }
        return important_knobs, unimportant_knobs

    def main(self) -> None:
        is_postgres_running = get_is_postgres_running()

        if is_postgres_running:
            st.write("Postgres is RUNNING")

            if st.button("Stop Postgres"):
                self.pg_conn.shutdown_postgres()
                st.rerun()

            with st.form("reconfig", clear_on_submit=True, enter_to_submit=False):
                knob = st.text_input("Knob", placeholder="Enter text here...")
                val = st.text_input("Value", placeholder="Enter text here...")
                submit_button = st.form_submit_button("Reconfigure")
            if submit_button:
                if knob != "" and val != "":
                    if "conf_changes" not in st.session_state:
                        st.session_state.conf_changes = dict()
                    
                    # By using st.session_state, we persist changes across st.rerun() (though not across reloading the browser).
                    st.session_state.conf_changes[knob] = val
                    self.pg_conn.restart_with_changes(st.session_state.conf_changes)
                    st.rerun()

            important_knobs, unimportant_knobs = self._get_categorized_system_knobs()
            with st.expander("Important knobs", expanded=True):
                st.write(important_knobs)

            with st.expander("Other knobs", expanded=False):
                st.write(unimportant_knobs)
        else:
            st.write("Postgres is STOPPED")

            if st.button("Start Postgres"):
                self.pg_conn.restore_pristine_snapshot()
                self.pg_conn.restart_postgres()
                st.rerun()


if __name__ == "__main__":
    demo = Demo()
    demo.main()
