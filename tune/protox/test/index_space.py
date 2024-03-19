import numpy as np
import yaml
import json
import unittest
from tune.protox.env.workload import Workload
from tune.protox.env.space.primitive_spaces import IndexSpace
from tune.protox.env.space.utils import check_subspace

class IndexSpaceTest(unittest.TestCase):
    @staticmethod
    def load(config_file="configs/benchmark/tpch.yaml", aux_type=True, aux_include=True):
        with open(config_file, "r") as f:
            benchmark_config = yaml.safe_load(f)
            benchmark_key = [k for k in benchmark_config.keys()][0]
            benchmark_config = benchmark_config[benchmark_key]
            benchmark_config["benchmark"] = benchmark_key

        w = Workload(
            tables=benchmark_config["tables"],
            attributes=benchmark_config["attributes"],
            query_spec=benchmark_config["query_spec"],
            pid=None,
            workload_timeout=0,
            workload_timeout_penalty=1.,
            logger=None,
        )

        i = IndexSpace(
            tables=benchmark_config["tables"],
            max_num_columns=benchmark_config["max_num_columns"],
            max_indexable_attributes=w.max_indexable(),
            seed=0,
            rel_metadata=benchmark_config["attributes"],
            attributes_overwrite=w.column_usages(),
            tbl_include_subsets=w.tbl_include_subsets,
            index_space_aux_type=aux_type,
            index_space_aux_include=aux_include,
            deterministic_policy=True,
        )
        return w, i


    def test_null_action(self):
        w, i = IndexSpaceTest.load()
        null_action = i.null_action()
        self.assertTrue(check_subspace(i, null_action))

        w, i = IndexSpaceTest.load(aux_type=False, aux_include=False)
        null_action = i.null_action()
        self.assertTrue(check_subspace(i, null_action))

    def test_sample(self):
        w, i = IndexSpaceTest.load(aux_type=False, aux_include=False)
        for _ in range(100):
            self.assertTrue(check_subspace(i, i.sample()))

    def test_sample_table(self):
        w, i = IndexSpaceTest.load(aux_type=False, aux_include=False)
        for _ in range(100):
            mask = {"table_idx": 2}
            ia = i.to_action(i.sample(mask))
            self.assertEqual(ia.tbl_name, "lineitem")

    def test_sample_table_col(self):
        w, i = IndexSpaceTest.load(aux_type=False, aux_include=False)
        for _ in range(100):
            mask = {"table_idx": 2, "col_idx": 1}
            ia = i.to_action(i.sample(mask))
            self.assertEqual(ia.tbl_name, "lineitem")
            self.assertEqual(ia.columns[0], "l_partkey")

    def test_neighborhood(self):
        w, i = IndexSpaceTest.load(aux_type=True, aux_include=True)
        _, isa = IndexSpaceTest.load(aux_type=False, aux_include=False)

        act = isa.sample(mask={"table_idx": 2, "col_idx": 1})
        act = (0, *act, np.zeros(i.max_inc_columns, dtype=np.float32))
        self.assertTrue(check_subspace(i, act))

        neighbors = i.policy.structural_neighbors(act)
        for n in neighbors:
            ia = i.to_action(n)
            self.assertEqual(n, ia.raw_repr)
            self.assertTrue(check_subspace(i, n))
