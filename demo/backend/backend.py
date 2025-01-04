import time
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
from gymlib.workspace import fully_resolve_path, make_standard_dbgym_workspace
from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


@app.route('/submit', methods=['POST'])
def submit():
    # data = request.json
    data = {"name": "John Doe"}
    time.sleep(5)
    return {
        "your_results": {
            "name": data["name"],
            "runtime": 8.5,
            "in_top_results": False,
            "rank": 18,
        },
        "top_results": [
            {"name": "Alice Doe", "runtime": 1.5},
            {"name": "John Doe", "runtime": 2.0},
            {"name": "Jane Doe", "runtime": 2.5},
            {"name": "Chris Doe", "runtime": 3.0},
            {"name": "Emily Doe", "runtime": 3.2},
            {"name": "Michael Doe", "runtime": 3.5},
            {"name": "Sarah Doe", "runtime": 4.0},
            {"name": "David Doe", "runtime": 4.5},
            {"name": "Laura Doe", "runtime": 5.0},
            {"name": "Tom Doe", "runtime": 5.5},
        ],
    }


class DemoBackend:
    def __init__(self):
        self.dbgym_workspace = make_standard_dbgym_workspace()
        self.pg_conn = PostgresConn(
            self.dbgym_workspace,
            DEFAULT_POSTGRES_PORT,
            fully_resolve_path(
                get_dbdata_tgz_symlink_path(
                    self.dbgym_workspace.dbgym_workspace_path, "job", DEFAULT_SCALE_FACTOR
                )
            ),
            self.dbgym_workspace.dbgym_tmp_path,
            fully_resolve_path(
                get_pgbin_symlink_path(self.dbgym_workspace.dbgym_workspace_path)
            ),
            None,
        )
        self.workload = Workload(
            self.dbgym_workspace,
            fully_resolve_path(
                get_workload_symlink_path(
                    self.dbgym_workspace.dbgym_workspace_path,
                    "job",
                    DEFAULT_SCALE_FACTOR,
                    get_workload_suffix("job", query_subset="demo"),
                )
            ),
        )
        self.pg_conn.restore_pristine_snapshot()
        self.pg_conn.restart_postgres()

    def time_workload(self):
        return self.pg_conn.time_workload(self.workload)

    def shutdown_postgres(self):
        self.pg_conn.shutdown_postgres()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=15721)