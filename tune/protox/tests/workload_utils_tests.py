import unittest
import pglast

from tune.protox.env.pglast3_workload_utils import QueryType, extract_sqltypes

class WorkloadUtilsTests(unittest.TestCase):
    TPCH_Q1 = '''
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
'''

    def test_extract_sqltypes(self):
        sql = WorkloadUtilsTests.TPCH_Q1
        stmts = pglast.Node(pglast.parse_sql(sql))
        pid = 0
        sqltypes = extract_sqltypes(stmts, pid)
        self.assertEqual(len(sqltypes), 1)
        self.assertTrue(type(sqltypes[0]) is tuple and len(sqltypes[0]) == 2)
        self.assertEqual(sqltypes[0][0], QueryType.SELECT)


if __name__ == '__main__':
    unittest.main()