import yaml
import json
import unittest
from pathlib import Path

from tune.protox.env.workload import Workload
from tune.protox.env.space.primitive_space import IndexSpace


class WorkloadTests(unittest.TestCase):
    @staticmethod
    def load(config_file: str, workload_path: Path):
        # don't call open_and_save() because this is a unittest
        with open(config_file, "r") as f:
            benchmark_config = yaml.safe_load(f)
            benchmark_key = [k for k in benchmark_config.keys()][0]
            benchmark_config = benchmark_config[benchmark_key]
            benchmark_config["benchmark"] = benchmark_key

        w = Workload(
            None,
            tables=benchmark_config["tables"],
            attributes=benchmark_config["attributes"],
            query_spec=benchmark_config["query_spec"],
            workload_path=workload_path,
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
            rel_metadata=w.column_usages(),
            attributes_overwrite=w.column_usages(),
            tbl_include_subsets={},
            index_space_aux_type=True,
            index_space_aux_include=True,
            deterministic_policy=True,
        )
        return w, i

    def diff_classmapping(self, ref, target):
        for k, v in ref.items():
            self.assertTrue(k in target, msg=f"{k} is missing.")
            self.assertTrue(v == target[k])

    def test_tpch(self):
        with open("tune/protox/tests/unittest_ref_models/ref_tpch_model.txt", "r") as f:
            ref = json.load(f)["class_mapping"]
            ref = {
                (v["relname"], v["ord_column"]): int(k)
                for k, v in ref.items()
            }

        w, i = WorkloadTests.load("tune/protox/tests/unittest_benchmark_configs/unittest_tpch.yaml", Path("tune/protox/tests/unittest_tpch_dir"))
        self.assertEqual(i.class_mapping, ref)

    def test_job(self):
        # don't call open_and_save() because this is a unittest
        with open("tune/protox/tests/unittest_ref_models/ref_job_full_model.txt", "r") as f:
            ref = json.load(f)["class_mapping"]
            ref = {
                (v["relname"], v["ord_column"]): int(k)
                for k, v in ref.items()
            }

        w, i = WorkloadTests.load("tune/protox/tests/unittest_benchmark_configs/unittest_job_full.yaml", Path("tune/protox/tests/unittest_job_full_dir"))
        self.assertEqual(i.class_mapping, ref)

    def test_dsb(self):
        # don't call open_and_save() because this is a unittest
        with open("tune/protox/tests/unittest_ref_models/ref_dsb_model.txt", "r") as f:
            ref = json.load(f)["class_mapping"]
            ref = {
                (v["relname"], v["ord_column"]): int(k)
                for k, v in ref.items()
            }

        w, i = WorkloadTests.load("tune/protox/tests/unittest_benchmark_configs/unittest_dsb.yaml", Path("tune/protox/tests/unittest_dsb_dir"))
        self.diff_classmapping(ref, i.class_mapping)

    def test_tpcc(self):
        # don't call open_and_save() because this is a unittest
        with open("tune/protox/tests/unittest_ref_models/ref_tpcc_model.txt", "r") as f:
            ref = json.load(f)["class_mapping"]
            ref = {
                (v["relname"], v["ord_column"]): int(k)
                for k, v in ref.items()
            }

        w, i = WorkloadTests.load("tune/protox/tests/unittest_benchmark_configs/unittest_tpcc.yaml", Path("tune/protox/tests/unittest_tpcc_dir"))
        self.assertEqual(i.class_mapping, ref)
