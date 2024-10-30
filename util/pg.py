"""
There are multiple parts of the codebase which interact with Postgres. This file contains helpers common to all those parts.
"""

from pathlib import Path
from typing import Any

import pglast
import psutil
import psycopg
import sqlalchemy
from sqlalchemy import create_engine, text

from util.workspace import DBGymConfig, open_and_save

DBGYM_POSTGRES_USER = "dbgym_user"
DBGYM_POSTGRES_PASS = "dbgym_pass"
DBGYM_POSTGRES_DBNAME = "dbgym"
DEFAULT_POSTGRES_DBNAME = "postgres"
DEFAULT_POSTGRES_PORT = 5432
SHARED_PRELOAD_LIBRARIES = "boot,pg_hint_plan,pg_prewarm"


def sqlalchemy_conn_execute(
    conn: sqlalchemy.Connection, sql: str
) -> sqlalchemy.engine.CursorResult[Any]:
    return conn.execute(text(sql))


def sql_file_queries(dbgym_cfg: DBGymConfig, filepath: Path) -> list[str]:
    with open_and_save(dbgym_cfg, filepath) as f:
        lines: list[str] = []
        for line in f:
            if line.startswith("--"):
                continue
            if len(line.strip()) == 0:
                continue
            lines.append(line)
        queries_str = "".join(lines)
        queries: list[str] = pglast.split(queries_str)
        return queries


def sql_file_execute(
    dbgym_cfg: DBGymConfig, conn: sqlalchemy.Connection, filepath: Path
) -> None:
    for sql in sql_file_queries(dbgym_cfg, filepath):
        sqlalchemy_conn_execute(conn, sql)


# The reason pgport is an argument is because when doing agnet HPO, we want to run multiple instances of Postgres
#   at the same time. In this situation, they need to have different ports
def get_connstr(pgport: int = DEFAULT_POSTGRES_PORT, use_psycopg: bool = True) -> str:
    connstr_suffix = f"{DBGYM_POSTGRES_USER}:{DBGYM_POSTGRES_PASS}@localhost:{pgport}/{DBGYM_POSTGRES_DBNAME}"
    # use_psycopg means whether or not we use the psycopg.connect() function
    # counterintuively, you *don't* need psycopg in the connection string if you *are*
    #   using the psycopg.connect() function
    connstr_prefix = "postgresql" if use_psycopg else "postgresql+psycopg"
    return connstr_prefix + "://" + connstr_suffix


def get_kv_connstr(pgport: int = DEFAULT_POSTGRES_PORT) -> str:
    return f"host=localhost port={pgport} user={DBGYM_POSTGRES_USER} password={DBGYM_POSTGRES_PASS} dbname={DBGYM_POSTGRES_DBNAME}"


def create_psycopg_conn(pgport: int = DEFAULT_POSTGRES_PORT) -> psycopg.Connection[Any]:
    connstr = get_connstr(use_psycopg=True, pgport=pgport)
    psycopg_conn = psycopg.connect(connstr, autocommit=True, prepare_threshold=None)
    return psycopg_conn


def create_sqlalchemy_conn(
    pgport: int = DEFAULT_POSTGRES_PORT,
) -> sqlalchemy.Connection:
    connstr = get_connstr(use_psycopg=False, pgport=pgport)
    engine: sqlalchemy.Engine = create_engine(
        connstr,
        execution_options={"isolation_level": "AUTOCOMMIT"},
    )
    return engine.connect()


def get_is_postgres_running() -> bool:
    """
    This is often used in assertions to ensure that Postgres isn't running before we
    execute some code.

    I intentionally do not have a function that forcefully *stops* all Postgres instances.
    This is risky because it could accidentally stop instances it wasn't supposed (e.g.
    Postgres instances run by other users on the same machine).

    Stopping Postgres instances is thus a responsibility of the human to take care of.
    """
    return len(get_running_postgres_ports()) > 0


def get_running_postgres_ports() -> list[int]:
    """
    Returns a list of all ports on which Postgres is currently running.

    There are ways to check with psycopg/sqlalchemy. However, I chose to check using
    psutil to keep it as simple as possible and orthogonal to how connections work.
    """
    running_ports = []

    for conn in psutil.net_connections(kind="inet"):
        if conn.status == "LISTEN":
            try:
                proc = psutil.Process(conn.pid)
                if proc.name() == "postgres":
                    running_ports.append(conn.laddr.port)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    return running_ports
