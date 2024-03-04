from enum import unique, Enum
import time
import pglast
import pglast.ast
from pglast import stream
from pglast.visitors import Visitor, Continue
import logging
import time
from collections import Counter
import psycopg
from psycopg.errors import QueryCanceled
import math

@unique
class QueryType(Enum):
    UNKNOWN = -1
    SELECT = 0
    CREATE_VIEW = 1
    DROP_VIEW = 2
    INS_UPD_DEL = 3


def parse_access_method(explain_data):
    def recurse(data):
        sub_data = {}
        if "Plans" in data:
            for p in data["Plans"]:
                sub_data.update(recurse(p))
        elif "Plan" in data:
            sub_data.update(recurse(data["Plan"]))

        if "Alias" in data:
            sub_data[data["Alias"]] = data["Node Type"]
        return sub_data
    return recurse(explain_data)


def force_statement_timeout(connection, timeout_ms):
    retry = True
    while retry:
        retry = False
        try:
            connection.execute(f"SET statement_timeout = {timeout_ms}")
        except QueryCanceled:
            retry = True

def time_query(prefix, connection, qid, query, timeout):
    has_timeout = False
    has_explain = "EXPLAIN" in query
    explain_data = None

    try:
        start_time = time.time()
        cursor = connection.execute(query)
        qid_runtime = (time.time() - start_time) * 1e6

        if has_explain:
            c = [c for c in cursor][0][0][0]
            assert "Execution Time" in c
            qid_runtime = float(c["Execution Time"]) * 1e3
            explain_data = c

        logging.debug(f"{prefix} {qid} evaluated in {qid_runtime/1e6}")

    except QueryCanceled:
        logging.debug(f"{prefix} {qid} exceeded evaluation timeout {timeout}")
        qid_runtime = timeout * 1e6
        has_timeout = True
    except Exception as e:
        assert False, print(e)
    # qid_runtime is in microseconds.
    return qid_runtime, has_timeout, explain_data


def acquire_metrics_around_query(prefix, env_spec, connection, qid, query, qtimeout, metrics=False):
    args = {"connection": connection}
    force_statement_timeout(connection, 0)
    if metrics:
        initial_metrics = env_spec.observation_space.construct_online(**args)

    if qtimeout is not None and qtimeout > 0:
        force_statement_timeout(connection, qtimeout * 1000)

    qid_runtime, main_timeout, explain_data = time_query(prefix, connection, qid, query, qtimeout)

    # Wipe the statement timeout.
    force_statement_timeout(connection, 0)
    if metrics:
        final_metrics = env_spec.observation_space.construct_online(**args)
        diff = env_spec.observation_space.construct_metric_delta(initial_metrics, final_metrics)
    else:
        diff = None
    # qid_runtime is in microseconds.
    return diff, qid_runtime, main_timeout, explain_data


def execute_serial_variations(env_spec, connection, timeout, logger, qid, query, runs, real_knobs):
    # Whether need metric.
    need_metric = env_spec.observation_space.metrics()
    # Initial timeout.
    timeout_limit = timeout
    # Best run invocation.
    best_metric, best_time, best_timeout, best_explain_data, runs_idx = None, None, True, None, None

    for prefix, pqk in runs:
        # Attach the specific per-query knobs.
        pqk_query = "/*+ " + " ".join([knob.resolve_per_query_knob(value, all_knobs=real_knobs) for (knob, value) in pqk]) + " */" + query
        # Log the query plan.
        pqk_query = "EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF) " + pqk_query

        # Log out the knobs that we are using.
        pqkk = [(knob.name(), val) for (knob, val) in pqk]
        logging.debug(f"{prefix} executing with {pqkk}")

        metric, runtime, did_timeout, explain_data = acquire_metrics_around_query(
            prefix=prefix,
            env_spec=env_spec,
            connection=connection,
            qid=qid,
            query=pqk_query,
            qtimeout=timeout_limit,
            metrics=need_metric)

        if not did_timeout:
            new_timeout_limit = math.ceil(runtime / 1e3) / 1.e3
            if new_timeout_limit < timeout_limit:
                timeout_limit = new_timeout_limit

        if best_time is None or runtime < best_time:
            best_metric = metric
            best_time = runtime
            best_timeout = did_timeout
            best_explain_data = explain_data
            runs_idx = (prefix, pqk)

        if logger is not None:
            logger.record(f"instr_time/{prefix}", runtime / 1e6)

    return best_metric, best_time, best_timeout, best_explain_data, runs_idx


def traverse(stmt):
    '''
    Trying to mimic the .traverse() function pglast v3 in pglast v6
    For context, we switched from pglast v3 to pglast v6
    '''
    visitor = Visitor()
    generator = visitor.iterate(stmt)

    try:
        item = generator.send(None)
        yield item
    except StopIteration:
        return

    while True:
        try:
            item = generator.send(Continue)
            yield item
        except StopIteration:
            return


def extract_aliases(stmts):
    # Extract the aliases.
    aliases = {}
    ctes = set()
    for stmt in stmts:
        for _, node in traverse(stmt):
            if isinstance(node, pglast.ast.Node):
                if isinstance(node, pglast.ast.CommonTableExpr):
                    ctes.add(node.ctename)
                elif isinstance(node, pglast.ast.RangeVar):
                    ft = node
                    relname = ft.relname
                    # TODO(phw2): convert to pglast v6
                    # if stmt.stmt["node_tag"] == "ViewStmt":
                    #     if node == stmt.stmt.view:
                    #         continue

                    alias = ft.relname if (ft.alias is None or ft.alias.aliasname is None or ft.alias.aliasname == "") else ft.alias.aliasname
                    if relname not in aliases:
                        aliases[relname] = []
                    if alias not in aliases[relname]:
                        aliases[relname].append(alias)
                    #else:
                    #    logging.warn(f"Squashing {relname} {alias} on {sql_file}")
    aliases = {k:v for k,v in aliases.items() if k not in ctes}
    return aliases

def extract_sqltypes(stmts, pid):
    sqls = []
    for stmt in stmts:
        sql_type = QueryType.UNKNOWN
        if isinstance(stmt, pglast.ast.RawStmt) and isinstance(stmt.stmt, pglast.ast.SelectStmt):
            sql_type = QueryType.SELECT
        elif isinstance(stmt, pglast.ast.RawStmt) and isinstance(stmt.stmt, pglast.ast.ViewStmt):
            sql_type = QueryType.CREATE_VIEW
        elif isinstance(stmt, pglast.ast.RawStmt) and isinstance(stmt.stmt, pglast.ast.DropStmt):
            drop_ast = stmt.stmt
            if drop_ast.removeType == pglast.enums.parsenodes.ObjectType.OBJECT_VIEW:
                sql_type = QueryType.DROP_VIEW
        elif isinstance(stmt, pglast.ast.RawStmt) and any([
            isinstance(stmt.stmt, pglast.ast.InsertStmt),
            isinstance(stmt.stmt, pglast.ast.UpdateStmt),
            isinstance(stmt.stmt, pglast.ast.DeleteStmt),
        ]):
            sql_type = QueryType.INS_UPD_DEL

        q = stream.RawStream()(stmt)
        if pid is not None and "pid" in q:
            q = q.replace("pid", str(pid))
        elif pid is not None and "PID" in q:
            q = q.replace("PID", str(pid))

        sqls.append((sql_type, q))
    return sqls

def extract_columns(stmt, tables, all_attributes, query_aliases):
    tbl_col_usages = {t: set() for t in tables}
    def traverse_extract_columns(alias_set, node, update=True):
        if node is None:
            return []

        columns = []
        for _, expr in traverse(node):
            if isinstance(expr, pglast.ast.Node) and isinstance(expr, pglast.ast.ColumnRef):
                if len(expr.fields) == 2:
                    tbl, col = expr.fields[0], expr.fields[1]
                    assert isinstance(tbl, pglast.ast.String) and isinstance(col, pglast.ast.String)

                    tbl = tbl.sval
                    col = col.sval
                    if tbl in alias_set and (tbl in tbl_col_usages or alias_set[tbl] in tbl_col_usages):
                        tbl = alias_set[tbl]
                        if update:
                            tbl_col_usages[tbl].add(col)
                        else:
                            columns.append((tbl, col))
                elif isinstance(expr.fields[0], pglast.ast.String):
                    col = expr.fields[0].sval
                    if col in all_attributes:
                        for tbl in all_attributes[col]:
                            if tbl in alias_set.values():
                                if update:
                                    tbl_col_usages[tbl].add(col)
                                else:
                                    columns.append((tbl, col))
        return columns

    # This is the query column usage.
    all_refs = []
    for _, node in traverse(stmt):
        if isinstance(node, pglast.ast.Node):
            if isinstance(node, pglast.ast.SelectStmt):
                aliases = {}
                for relname, relalias in query_aliases.items():
                    for alias in relalias:
                        aliases[alias] = relname

                # We derive the "touched" columns only from the WHERE clause.
                traverse_extract_columns(aliases, node.whereClause)
                all_refs.extend(traverse_extract_columns(aliases, node, update=False))
    return tbl_col_usages, all_refs
