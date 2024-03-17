from typing import List
from pathlib import Path

import pglast
from sqlalchemy import Connection, Engine, Inspector, inspect, text
from sqlalchemy.engine import CursorResult


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
