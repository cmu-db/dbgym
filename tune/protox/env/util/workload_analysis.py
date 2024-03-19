from enum import Enum, unique
from typing import Iterator, Optional, Tuple

import pglast # type: ignore
from pglast import stream
from pglast.visitors import Continue, Visitor # type: ignore
from tune.protox.env.types import TableAliasMap, AttrTableListMap, TableColTuple, TableAttrSetMap, QueryType


def traverse(stmt: pglast.ast.Node) -> Iterator[pglast.ast.Node]:
    """
    Trying to mimic the .traverse() function pglast v3 in pglast v6
    For context, we switched from pglast v3 to pglast v6
    """
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


def extract_aliases(stmts: pglast.ast.Node) -> TableAliasMap:
    # Extract the aliases.
    aliases: TableAliasMap = TableAliasMap({})
    ctes = set()
    for stmt in stmts:
        for _, node in traverse(stmt):
            if isinstance(node, pglast.ast.Node):
                if isinstance(node, pglast.ast.CommonTableExpr):
                    ctes.add(node.ctename)
                elif isinstance(node, pglast.ast.RangeVar):
                    ft = node
                    relname = ft.relname

                    alias = (
                        ft.relname
                        if (
                            ft.alias is None
                            or ft.alias.aliasname is None
                            or ft.alias.aliasname == ""
                        )
                        else ft.alias.aliasname
                    )
                    if relname not in aliases:
                        aliases[relname] = []
                    if alias not in aliases[relname]:
                        aliases[relname].append(alias)
                    # else:
                    #    logging.warn(f"Squashing {relname} {alias} on {sql_file}")
    return TableAliasMap({k: v for k, v in aliases.items() if k not in ctes})


def extract_sqltypes(
    stmts: pglast.ast.Node, pid: Optional[int]
) -> list[Tuple[QueryType, str]]:
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


def extract_columns(
    stmt: pglast.ast.Node,
    tables: list[str],
    all_attributes: AttrTableListMap,
    query_aliases: TableAliasMap,
) -> Tuple[TableAttrSetMap, list[TableColTuple]]:
    tbl_col_usages: TableAttrSetMap = TableAttrSetMap({t: set() for t in tables})

    def traverse_extract_columns(
        alias_set: dict[str, str], node: pglast.ast.Node, update: bool = True
    ) -> list[TableColTuple]:
        if node is None:
            return []

        columns = []
        for _, expr in traverse(node):
            if isinstance(expr, pglast.ast.Node) and isinstance(
                expr, pglast.ast.ColumnRef
            ):
                if len(expr.fields) == 2:
                    tbl, col = expr.fields[0], expr.fields[1]
                    assert isinstance(tbl, pglast.ast.String) and isinstance(
                        col, pglast.ast.String
                    )

                    tbl = tbl.sval
                    col = col.sval
                    if tbl in alias_set and (
                        tbl in tbl_col_usages or alias_set[tbl] in tbl_col_usages
                    ):
                        tbl = alias_set[tbl]
                        if update:
                            tbl_col_usages[tbl].add(col)
                        else:
                            columns.append(TableColTuple((tbl, col)))
                elif isinstance(expr.fields[0], pglast.ast.String):
                    col = expr.fields[0].sval
                    if col in all_attributes:
                        for tbl in all_attributes[col]:
                            if tbl in alias_set.values():
                                if update:
                                    tbl_col_usages[tbl].add(col)
                                else:
                                    columns.append(TableColTuple((tbl, col)))
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
