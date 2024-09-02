from pathlib import Path
from typing import Any, List

import pglast
import psycopg
from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.engine import CursorResult

from misc.utils import DBGymConfig, open_and_save

DBGYM_POSTGRES_USER = "dbgym_user"
DBGYM_POSTGRES_PASS = "dbgym_pass"
DBGYM_POSTGRES_DBNAME = "dbgym"
DEFAULT_POSTGRES_DBNAME = "postgres"
DEFAULT_POSTGRES_PORT = 5432
SHARED_PRELOAD_LIBRARIES = "boot,pg_hint_plan,pg_prewarm"


def conn_execute(conn: Connection, sql: str) -> CursorResult[Any]:
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


def sql_file_execute(dbgym_cfg: DBGymConfig, conn: Connection, filepath: Path) -> None:
    for sql in sql_file_queries(dbgym_cfg, filepath):
        conn_execute(conn, sql)


# The reason pgport is an argument is because when doing agnet HPO, we want to run multiple instances of Postgres
#   at the same time. In this situation, they need to have different ports
def get_connstr(pgport: int = DEFAULT_POSTGRES_PORT, use_psycopg: bool=True) -> str:
    connstr_suffix = f"{DBGYM_POSTGRES_USER}:{DBGYM_POSTGRES_PASS}@localhost:{pgport}/{DBGYM_POSTGRES_DBNAME}"
    # use_psycopg means whether or not we use the psycopg.connect() function
    # counterintuively, you *don't* need psycopg in the connection string if you *are*
    #   using the psycopg.connect() function
    connstr_prefix = "postgresql" if use_psycopg else "postgresql+psycopg"
    return connstr_prefix + "://" + connstr_suffix


def create_conn(pgport: int = DEFAULT_POSTGRES_PORT, use_psycopg: bool=True) -> Connection:
    connstr = get_connstr(use_psycopg=use_psycopg, pgport=pgport)
    if use_psycopg:
        psycopg_conn = psycopg.connect(connstr, autocommit=True, prepare_threshold=None)
        engine = create_engine(connstr, creator=lambda : psycopg_conn)
        return engine.connect()
    else:
        engine = create_engine(
            connstr,
            execution_options={"isolation_level": "AUTOCOMMIT"},
        )
        return engine.connect()
