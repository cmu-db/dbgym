import unittest

import pglast

from tune.protox.env.util.workload_analysis import *


class WorkloadUtilsTests(unittest.TestCase):
    TPCH_TABLES = [
        "part",
        "partsupp",
        "lineitem",
        "orders",
        "supplier",
        "customer",
        "nation",
        "region",
    ]
    TPCH_ALL_ATTRIBUTES = {
        "r_regionkey": ["region"],
        "r_name": ["region"],
        "r_comment": ["region"],
        "n_nationkey": ["nation"],
        "n_name": ["nation"],
        "n_regionkey": ["nation"],
        "n_comment": ["nation"],
        "p_partkey": ["part"],
        "p_name": ["part"],
        "p_mfgr": ["part"],
        "p_brand": ["part"],
        "p_type": ["part"],
        "p_size": ["part"],
        "p_container": ["part"],
        "p_retailprice": ["part"],
        "p_comment": ["part"],
        "s_suppkey": ["supplier"],
        "s_name": ["supplier"],
        "s_address": ["supplier"],
        "s_nationkey": ["supplier"],
        "s_phone": ["supplier"],
        "s_acctbal": ["supplier"],
        "s_comment": ["supplier"],
        "ps_partkey": ["partsupp"],
        "ps_suppkey": ["partsupp"],
        "ps_availqty": ["partsupp"],
        "ps_supplycost": ["partsupp"],
        "ps_comment": ["partsupp"],
        "c_custkey": ["customer"],
        "c_name": ["customer"],
        "c_address": ["customer"],
        "c_nationkey": ["customer"],
        "c_phone": ["customer"],
        "c_acctbal": ["customer"],
        "c_mktsegment": ["customer"],
        "c_comment": ["customer"],
        "o_orderkey": ["orders"],
        "o_custkey": ["orders"],
        "o_orderstatus": ["orders"],
        "o_totalprice": ["orders"],
        "o_orderdate": ["orders"],
        "o_orderpriority": ["orders"],
        "o_clerk": ["orders"],
        "o_shippriority": ["orders"],
        "o_comment": ["orders"],
        "l_orderkey": ["lineitem"],
        "l_partkey": ["lineitem"],
        "l_suppkey": ["lineitem"],
        "l_linenumber": ["lineitem"],
        "l_quantity": ["lineitem"],
        "l_extendedprice": ["lineitem"],
        "l_discount": ["lineitem"],
        "l_tax": ["lineitem"],
        "l_returnflag": ["lineitem"],
        "l_linestatus": ["lineitem"],
        "l_shipdate": ["lineitem"],
        "l_commitdate": ["lineitem"],
        "l_receiptdate": ["lineitem"],
        "l_shipinstruct": ["lineitem"],
        "l_shipmode": ["lineitem"],
        "l_comment": ["lineitem"],
    }
    TPCH_Q1 = """
select
	l_returnflag,
	l_linestatus,
	sum(l_quantity) as sum_qty,
	sum(l_extendedprice) as sum_base_price,
	sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
	sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
	avg(l_quantity) as avg_qty,
	avg(l_extendedprice) as avg_price,
	avg(l_discount) as avg_disc,
	count(*) as count_order
from
	lineitem
where
	l_shipdate <= date '1998-12-01' - interval '80' day
group by
	l_returnflag,
	l_linestatus
order by
	l_returnflag,
	l_linestatus;
"""

    @staticmethod
    def pglast_parse(sql):
        return pglast.parse_sql(sql)

    def test_extract_aliases(self):
        sql = "select * from t1 as t1_alias; select * from t1;"
        stmts = WorkloadUtilsTests.pglast_parse(sql)
        aliases = extract_aliases(stmts)
        # if a table has more than one alias we have to do this more verbose assertion code
        #     to make it order invariant
        self.assertTrue("t1" in aliases and len(aliases) == 1)
        self.assertEqual(set(aliases["t1"]), set(["t1", "t1_alias"]))

    def test_extract_aliases_ignores_views_in_create_view(self):
        sql = "create view view1 (view1_c1) as select c1 from t1;"
        stmts = WorkloadUtilsTests.pglast_parse(sql)
        aliases = extract_aliases(stmts)
        # all tables have only one alias so we can do this simpler assertion code
        self.assertEqual(aliases, {"t1": ["t1"]})

    def test_extract_aliases_doesnt_ignore_views_that_are_used(self):
        sql = "create view view1 (view1_c1) as select c1 from t1; select * from view1;"
        stmts = WorkloadUtilsTests.pglast_parse(sql)
        aliases = extract_aliases(stmts)
        # all tables have only one alias so we can do this simpler assertion code
        self.assertEqual(aliases, {"t1": ["t1"], "view1": ["view1"]})

    def test_extract_sqltypes(self):
        sql = """
select * from t1;
update t1 set t1.c1 = 0 where t1.c1 = 1;
create or replace view view1 (view1_c1) as
    select c1
    from t1;
"""
        stmts = WorkloadUtilsTests.pglast_parse(sql)
        pid = 0
        sqltypes = extract_sqltypes(stmts, pid)

        expected_num_stmts = 3
        self.assertEqual(len(sqltypes), expected_num_stmts)
        for i in range(expected_num_stmts):
            self.assertTrue(type(sqltypes[i]) is tuple and len(sqltypes[i]) == 2)
        self.assertEqual(sqltypes[0][0], QueryType.SELECT)
        self.assertEqual(sqltypes[1][0], QueryType.INS_UPD_DEL)
        self.assertEqual(sqltypes[2][0], QueryType.CREATE_VIEW)

    def test_extract_columns(self):
        sql = WorkloadUtilsTests.TPCH_Q1
        tables = WorkloadUtilsTests.TPCH_TABLES
        all_attributes = WorkloadUtilsTests.TPCH_ALL_ATTRIBUTES
        stmts = WorkloadUtilsTests.pglast_parse(sql)
        aliases = extract_aliases(stmts)
        self.assertEqual(len(stmts), 1)
        stmt = stmts[0]
        tbl_col_usages, all_refs = extract_columns(
            stmt, tables, all_attributes, aliases
        )

        for table in tables:
            self.assertTrue(table in tbl_col_usages)
            if table == "lineitem":
                self.assertEqual(tbl_col_usages[table], {"l_shipdate"})
            else:
                self.assertEqual(tbl_col_usages[table], set())

        self.assertEqual(
            set(all_refs),
            set(
                [
                    ("lineitem", "l_returnflag"),
                    ("lineitem", "l_linestatus"),
                    ("lineitem", "l_returnflag"),
                    ("lineitem", "l_linestatus"),
                    ("lineitem", "l_returnflag"),
                    ("lineitem", "l_linestatus"),
                    ("lineitem", "l_quantity"),
                    ("lineitem", "l_extendedprice"),
                    ("lineitem", "l_extendedprice"),
                    ("lineitem", "l_discount"),
                    ("lineitem", "l_extendedprice"),
                    ("lineitem", "l_discount"),
                    ("lineitem", "l_tax"),
                    ("lineitem", "l_quantity"),
                    ("lineitem", "l_extendedprice"),
                    ("lineitem", "l_discount"),
                    ("lineitem", "l_shipdate"),
                ]
            ),
        )

    def test_extract_columns_with_cte(self):
        sql = """
with cte1 as (
    select t1.c1
    from t1
    where t1.c2 = 3
)
select *
from cte1;
"""
        tables = ["t1"]
        all_attributes = {"c1": "t1", "c2": "t1"}
        stmts = WorkloadUtilsTests.pglast_parse(sql)
        aliases = extract_aliases(stmts)
        self.assertEqual(len(stmts), 1)
        stmt = stmts[0]
        tbl_col_usages, all_refs = extract_columns(
            stmt, tables, all_attributes, aliases
        )

        self.assertEqual(tbl_col_usages, {"t1": {"c2"}})
        self.assertEqual(
            set(all_refs), set([("t1", "c1"), ("t1", "c2"), ("t1", "c1"), ("t1", "c2")])
        )


if __name__ == "__main__":
    unittest.main()
