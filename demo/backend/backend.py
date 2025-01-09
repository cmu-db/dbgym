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


def drop_all_indexes() -> None:
    num_indexes = 5
    for i in range(num_indexes):
        demo_backend.pg_conn.psql(f"DROP INDEX IF EXISTS index{i}")


@app.route("/submit", methods=["POST"])
def submit() -> dict[str, Any]:
    # data = request.json DEBUG
    with open("demo/backend/pgtune.json", "r") as f:
        data = json.load(f)

    # Set system knobs (requires database restart).
    demo_backend.pg_conn.restart_with_changes(data["sysknobs"])

    # Create indexes. # TODO: create this separately.
    drop_all_indexes()
    demo_backend.pg_conn.psql("CREATE INDEX index0 ON movie_companies (movie_id)")
    demo_backend.pg_conn.psql(
        "CREATE INDEX index1 ON movie_keyword (keyword_id) INCLUDE (movie_id)"
    )
    demo_backend.pg_conn.psql(
        "CREATE INDEX index2 ON movie_keyword (movie_id) INCLUDE (keyword_id)"
    )

    # Run multiple trials and take the average since it's so short.
    NUM_TRIALS = 1
    total_runtime_us: float = 0
    for _ in range(NUM_TRIALS):
        runtime_us, _ = demo_backend.time_workload()
        total_runtime_us += runtime_us
    average_runtime_us = total_runtime_us / NUM_TRIALS
    runtime_s = average_runtime_us / 1_000_000

    # Since restart_with_changes() is not additive (see its comment), we actually *don't* need to reset the system knobs.
    # Next time restart_with_changes() is called, it will make changes from Postgres's default values.

    # Drop all indexes.
    drop_all_indexes()

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

    def time_workload(self) -> tuple[float, int]:
        return self.pg_conn.time_workload(self.workload)

    def shutdown_postgres(self) -> None:
        self.pg_conn.shutdown_postgres()


demo_backend = DemoBackend()


# TODO: make backend not have to start postgres every time. assert job table if postgres is up

if __name__ == "__main__":
    # host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    # app.run(host=host, port=15721) DEBUG
    submit()
