import pickle
import json
import unittest
from pathlib import Path
from typing import Any, Tuple

import yaml

from tune.protox.env.space.primitive_space import IndexSpace
from tune.protox.env.types import TableAttrAccessSetsMap, TableColTuple
from tune.protox.env.workload import Workload


class WorkloadTests(unittest.TestCase):
    @staticmethod
    def build(config_fpath: Path, workload_path: Path) -> tuple[Workload, IndexSpace]:
        # don't call open_and_save() because this is a unittest
        with open(config_fpath, "r") as f:
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
            workload_timeout_penalty=1.0,
            artifact_manager=None,
        )

        i = IndexSpace(
            tables=benchmark_config["tables"],
            max_num_columns=benchmark_config["max_num_columns"],
            max_indexable_attributes=w.max_indexable(),
            seed=0,
            rel_metadata=w.column_usages(),
            attributes_overwrite=w.column_usages(),
            tbl_include_subsets=TableAttrAccessSetsMap({}),
            index_space_aux_type=True,
            index_space_aux_include=True,
            deterministic_policy=True,
        )
        return w, i
    
    def _test_workload(self, workload_name: str) -> None:
        # Build objects.
        tests_dpath = Path("tune/protox/tests")
        w, i = WorkloadTests.build(
            tests_dpath / f"unittest_benchmark_configs/unittest_{workload_name}.yaml",
            (tests_dpath / f"unittest_{workload_name}_dir").resolve()
        )

        # Load reference objects.
        ref_dpath = tests_dpath / "unittest_ref"
        ref_workload_fpath = ref_dpath / f"ref_{workload_name}_workload.pkl"
        ref_idxspace_fpath = ref_dpath / f"ref_{workload_name}_idxspace.pkl"
        with open(ref_workload_fpath, "rb") as f:
            ref_w: Workload = pickle.load(f)
        with open(ref_idxspace_fpath, "rb") as f:
            ref_i: IndexSpace = pickle.load(f)

        # Check various workload fields.
        self.assertEqual(w.column_usages(), ref_w.column_usages())

        # Check various idxspace mapping.
        self.assertEqual(i.class_mapping, ref_i.class_mapping)

        # # Uncomment this to "update" the reference objects.
        # with open(ref_workload_fpath, "wb") as f:
        #     pickle.dump(w, f)
        # with open(ref_idxspace_fpath, "wb") as f:
        #     pickle.dump(i, f)

    def test_tpch(self) -> None:
        self._test_workload("tpch")

    def test_jobfull(self) -> None:
        self._test_workload("jobfull")

    def test_dsb(self) -> None:
        self._test_workload("dsb")

    def test_tpcc(self) -> None:
        self._test_workload("tpcc")


if __name__ == "__main__":
    unittest.main()
