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
    def load(config_file: str, workload_path: Path) -> tuple[Workload, IndexSpace]:
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
        w, i = WorkloadTests.load(
            f"tune/protox/tests/unittest_benchmark_configs/unittest_{workload_name}.yaml",
            Path(f"tune/protox/tests/unittest_{workload_name}_dir").resolve(),
        )

        # Check class mapping
        with open(f"tune/protox/tests/unittest_ref_models/ref_{workload_name}_model.txt", "r") as f:
            ref_class_mapping = json.load(f)["class_mapping"]
            # Reformat it so that it's the same format as in the index space
            ref_class_mapping = {(v["relname"], v["ord_column"]): int(k) for k, v in ref_class_mapping.items()}
        self.assertEqual(i.class_mapping, ref_class_mapping)

        # Check column usages
        print(w.column_usages())

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
