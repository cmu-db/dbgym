import unittest
import pglast

import tune.protox.env.pglast3_workload_utils as pl3_utils

class WorkloadUtilsTests(unittest.TestCase):
    @staticmethod
    def pglast3_parse(sql):
        return pglast.Node(pglast.parse_sql(sql))
    

    @staticmethod
    def pglast_parse(sql):
        # return pglast.parse_sql(sql)
        return WorkloadUtilsTests.pglast3_parse(sql)


    def test_extract_aliases(self):
        sql = "select * from t1 as t1_alias; select * from t1;"
        stmts = WorkloadUtilsTests.pglast_parse(sql)
        aliases = pl3_utils.extract_aliases(stmts)
        self.assertTrue('t1' in aliases and len(aliases) == 1)
        self.assertEqual(set(aliases['t1']), set(['t1', 't1_alias']))


    def test_extract_sqltypes(self):
        sql = '''
select * from t1;
update t1 set t1.c1 = 0 where t1.c1 = 1;
create or replace view view1 (view1_c1) as
    select c1
    from t1;
'''
        stmts = WorkloadUtilsTests.pglast_parse(sql)
        pid = 0
        sqltypes = pl3_utils.extract_sqltypes(stmts, pid)

        expected_num_stmts = 3
        self.assertEqual(len(sqltypes), expected_num_stmts)
        for i in range(expected_num_stmts):
            self.assertTrue(type(sqltypes[i]) is tuple and len(sqltypes[i]) == 2)
        self.assertEqual(sqltypes[0][0], pl3_utils.QueryType.SELECT)
        self.assertEqual(sqltypes[1][0], pl3_utils.QueryType.INS_UPD_DEL)
        self.assertEqual(sqltypes[2][0], pl3_utils.QueryType.CREATE_VIEW)
    

if __name__ == '__main__':
    unittest.main()