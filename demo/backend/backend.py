import json
from pathlib import Path
from typing import Any

from flask import Flask, request
from flask_cors import CORS
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

app = Flask(__name__)
CORS(app)

# This is the max # of indexes the frontend can submit.
# The frontend should do its own checks to prevent this.
MAX_NUM_INDEXES = 10


def drop_indexes() -> None:
    for i in range(MAX_NUM_INDEXES):
        demo_backend.pg_conn.psql(f"DROP INDEX IF EXISTS index{i}")


@app.route("/submit", methods=["POST"])
def submit() -> dict[str, Any]:
    # data = request.json DEBUG
    with open("demo/backend/protox.json", "r") as f:
        data = json.load(f)

    # Set system knobs (requires database restart).
    demo_backend.pg_conn.restart_with_changes(data["sysknobs"])

    # Create indexes.
    # Drop first to avoid index name conflicts.
    assert len(data["indexes"]) <= MAX_NUM_INDEXES
    drop_indexes()
    for i, index_config in enumerate(data["indexes"]):
        includes_str = "" if index_config["include"] is None else f" INCLUDE ({index_config['include']})"
        create_index_sql = f"CREATE INDEX index{i} ON {index_config['table']} USING {index_config['type']} ({index_config['column']}){includes_str}"
        demo_backend.pg_conn.psql(create_index_sql)

    # Translate query knobs.
    qknobs = {
        "Q1a": data["qknobs"]["q1"],
        "Q2a": data["qknobs"]["q2"],
        "Q4a": data["qknobs"]["q3"],
    }

    # Run workload.
    runtime_us, _ = demo_backend.time_workload(qknobs=qknobs)
    runtime_s = runtime_us / 1_000_000

    # Since restart_with_changes() is not additive (see its comment), we actually *don't* need to reset the system knobs.
    # Next time restart_with_changes() is called, it will make changes from Postgres's default values.

    # Drop all indexes.
    drop_indexes()

    print(f"Runtime: {runtime_s:.3f}s")
    return {
        "runtime": runtime_s,
        "rank": 2,
    }


@app.route("/leaderboard", methods=["GET"])
def get_leaderboard() -> dict[str, Any]:
    return {
        "top_results": [
            {"name": "Alice Doe", "runtime": 1.5},
            {"name": "John Doe", "runtime": 2.0},
            {"name": "Bob Smith", "runtime": 2.5},
            {"name": "Charlie Brown", "runtime": 3.0},
            {"name": "Diana Prince", "runtime": 3.5},
            {"name": "Ethan Hunt", "runtime": 4.0},
            {"name": "Fiona Gallagher", "runtime": 4.5},
            {"name": "George Costanza", "runtime": 5.0},
            {"name": "Hannah Montana", "runtime": 5.5},
            {"name": "Ivy League", "runtime": 6.0},
        ]
    }


class DemoBackend:
    def __init__(self) -> None:
        self.dbgym_workspace = make_standard_dbgym_workspace()
        self.pg_conn = PostgresConn(
            self.dbgym_workspace,
            DEFAULT_POSTGRES_PORT,
            fully_resolve_path(
                get_dbdata_tgz_symlink_path(
                    self.dbgym_workspace.dbgym_workspace_path,
                    "job",
                    DEFAULT_SCALE_FACTOR,
                )
            ),
            Path("/mnt/nvme0n1/phw2"),
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

        # We put dbdata in a place where it won't get deleted every time.
        # Thus, we only need to do this once ever. It usually takes 44s to restore so it's good to cache it.
        if not self.pg_conn.dbdata_path.exists():
            self.pg_conn.restore_pristine_snapshot()

        self.pg_conn.restart_postgres()

    def time_workload(self, *args, **kwargs) -> tuple[float, int]:
        return self.pg_conn.time_workload(self.workload, *args, **kwargs)

    def shutdown_postgres(self) -> None:
        self.pg_conn.shutdown_postgres()


demo_backend = DemoBackend()


# TODO: make backend not have to start postgres every time. assert job table if postgres is up

if __name__ == "__main__":
    # host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    # app.run(host=host, port=15721) DEBUG
    submit()
