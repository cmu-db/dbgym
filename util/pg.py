from pathlib import Path
from typing import List
import pglast
from sqlalchemy import Connection, Engine, text, create_engine
from sqlalchemy.engine import CursorResult
import psycopg

from misc.utils import DBGymConfig

DBGYM_POSTGRES_DBNAME = "dbgym"


def conn_execute(conn: Connection, sql: str) -> CursorResult:
    return conn.execute(text(sql))


def sql_file_queries(filepath: Path) -> List[str]:
    with open(filepath) as f:
        lines: list[str] = []
        for line in f:
            if line.startswith("--"):
                continue
            if len(line.strip()) == 0:
                continue
            lines.append(line)
        queries = "".join(lines)
        return pglast.split(queries)


def sql_file_execute(conn: Connection, filepath: Path) -> None:
    for sql in sql_file_queries(filepath):
        conn_execute(conn, sql)


def get_connstr(dbgym_cfg: DBGymConfig, use_psycopg=True) -> str:
    pguser = dbgym_cfg.root_yaml["postgres_user"]
    pgpass = dbgym_cfg.root_yaml["postgres_pass"]
    pgport = dbgym_cfg.root_yaml["postgres_port"]
    connstr_suffix = f"{pguser}:{pgpass}@localhost:{pgport}/{DBGYM_POSTGRES_DBNAME}"
    # use_psycopg means whether or not we use the psycopg.connect() function
    # counterintuively, you *don't* need psycopg in the connection string if you *are*
    #   using the psycopg.connect() function
    connstr_prefix = "postgresql" if use_psycopg else "postgresql+psycopg"
    return connstr_prefix + "://" + connstr_suffix


def create_conn(dbgym_cfg: DBGymConfig, use_psycopg=True) -> Connection:
    connstr = get_connstr(dbgym_cfg, use_psycopg=use_psycopg)
    if use_psycopg:
        return psycopg.connect(
            connstr, autocommit=True, prepare_threshold=None
        )
    else:
        engine: Engine = create_engine(
            connstr,
            execution_options={"isolation_level": "AUTOCOMMIT"},
        )
        return engine.connect()
