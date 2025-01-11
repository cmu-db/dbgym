import json
from pathlib import Path
import sys
import sqlite3
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

# This is the max # of indexes that can be submitted.
# We'll throw an error if this is exceeded.
# The frontend should prevent this from happening.
MAX_NUM_INDEXES = 5


def drop_indexes() -> None:
    for i in range(MAX_NUM_INDEXES):
        demo_backend.pg_conn.psql(f"DROP INDEX IF EXISTS index{i}")


@app.route("/submit", methods=["POST"])
def submit() -> dict[str, Any]:
    return process_submission(request.json)


def process_submission(data: dict[str, Any]) -> dict[str, Any]:
    # Set system knobs (requires database restart).
    # sysknobs will not be in data if the user ran out of time before getting to the sysknobs page.
    if "sysknobs" in data:
        demo_backend.pg_conn.restart_with_changes(data["sysknobs"])

    # Create indexes.
    drop_indexes()
    if "indexes" in data:
        # Drop first to avoid index name conflicts.
        assert len(data["indexes"]) <= MAX_NUM_INDEXES
        for i, index_config in enumerate(data["indexes"]):
            includes_str = "" if index_config["include"] is None else f" INCLUDE ({index_config['include']})"
            create_index_sql = f"CREATE INDEX index{i} ON {index_config['table']} ({index_config['column']}){includes_str}"
            demo_backend.pg_conn.psql(create_index_sql)

    # Set up query knobs.
    qknobs = dict()
    if "qknobs" in data:
        qknobs = {
            "Q1a": data["qknobs"]["q1"],
            "Q2a": data["qknobs"]["q2"],
            "Q4a": data["qknobs"]["q3"],
        }

    # Run workload.
    total_runtime_us = 0
    trials = 3
    for _ in range(trials):
        runtime_us, _ = demo_backend.time_workload(qknobs=qknobs)
        total_runtime_us += runtime_us
    runtime_s = total_runtime_us / trials / 1_000_000

    # Add to leaderboard if the user has a name.
    best_runtime_s = None
    if data["welcomeData"]["name"]:
        leaderboard = Leaderboard()
        # We create a new leaderboard object each time because SQLite requires you to create the object in the same thread you use it in.
        best_runtime_s = leaderboard.update_user_best_runtime(name=data["welcomeData"]["name"], runtime=runtime_s)
        leaderboard.close()

    # Since restart_with_changes() is not additive (see its comment), we actually *don't* need to reset the system knobs.
    # Next time restart_with_changes() is called, it will make changes from Postgres's default values.

    # Drop all indexes.
    drop_indexes()

    return {
        "runtime": runtime_s,
        "best_runtime": best_runtime_s,
    }


@app.route("/leaderboard", methods=["GET"])
def get_leaderboard() -> dict[str, Any]:
    leaderboard = Leaderboard()
    top_results = leaderboard.get_top_users(10)
    leaderboard.close()
    return {
        "top_results": top_results,
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
            Path(".."),
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


class Leaderboard:
    def __init__(self):
        # leaderboard_dbname is set in if __name__ == "__main__"
        # Connect to database (creates it if it doesn't exist)
        self.conn = sqlite3.connect(leaderboard_dbname)

        # Create table if it doesn't exist
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                runtime REAL NOT NULL
            )
        ''')
        self.conn.commit()
    
    def update_user_best_runtime(self, name: str, runtime: float) -> float:
        self.cursor.execute('''
            INSERT INTO users (name, runtime)
            VALUES (?, ?)
            ON CONFLICT(name) DO UPDATE SET runtime = excluded.runtime
            WHERE excluded.runtime < users.runtime
        ''', (name, runtime))
        self.conn.commit()
        
        # Fetch the best runtime for the user after the update
        self.cursor.execute('SELECT runtime FROM users WHERE name = ?', (name,))
        best_runtime = self.cursor.fetchone()[0]
        return best_runtime

    def get_top_users(self, limit: int) -> list[dict[str, float]]:
        """Fetch the top X users based on their runtime."""
        self.cursor.execute('''
            SELECT name, runtime FROM users
            ORDER BY runtime ASC
            LIMIT ?
        ''', (limit,))
        return [{"name": name, "runtime": runtime} for name, runtime in self.cursor.fetchall()]
    
    def close(self) -> None:
        # Always close the connection when done
        self.conn.close()


demo_backend = DemoBackend()


# TODO: make backend not have to start postgres every time. assert job table if postgres is up

if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    leaderboard_dbname = sys.argv[2] if len(sys.argv) > 2 else "leaderboard.db"

    do_process_anchors = True
    if do_process_anchors:
        for name in ["protox", "pgtune_nuc"]:
            with open(f"demo/backend/{name}.json", "r") as f:
                data = json.load(f)
                process_submission(data)

    app.run(host=host, port=15721)
