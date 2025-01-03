from gymlib.infra_paths import (
    DEFAULT_SCALE_FACTOR,
    get_dbdata_tgz_symlink_path,
    get_pgbin_symlink_path,
    get_workload_suffix,
    get_workload_symlink_path,
)
from gymlib.pg import DEFAULT_POSTGRES_PORT
from gymlib.pg_conn import PostgresConn
from gymlib.workload import Workload
from gymlib.workspace import make_standard_dbgym_workspace

if __name__ == "__main__":
    dbgym_workspace = make_standard_dbgym_workspace()
    pg_conn = PostgresConn(
        dbgym_workspace,
        DEFAULT_POSTGRES_PORT,
        get_dbdata_tgz_symlink_path(
            dbgym_workspace.dbgym_workspace_path, "job", DEFAULT_SCALE_FACTOR
        ),
        dbgym_workspace.dbgym_tmp_path,
        get_pgbin_symlink_path(dbgym_workspace.dbgym_workspace_path),
        None,
    )
    workload = Workload(
        dbgym_workspace,
        get_workload_symlink_path(
            dbgym_workspace.dbgym_workspace_path,
            "job",
            DEFAULT_SCALE_FACTOR,
            get_workload_suffix("job", query_subset="demo"),
        ),
    )
