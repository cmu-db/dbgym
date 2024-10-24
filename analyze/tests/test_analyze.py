import unittest
from pathlib import Path

from analyze.cli import get_total_instr_time_event, tboard_to_minimal_json


class AnalyzeTests(unittest.TestCase):
    def test_tfevents(self) -> None:
        tfevents_path = Path("analyze/tests/unittest_analysis_files/out.tfevents")
        minimal_json = tboard_to_minimal_json(tfevents_path)
        self.assertAlmostEqual(
            get_total_instr_time_event(minimal_json, r".*PostgresEnv_reset$"), 8.0046994
        )
        self.assertAlmostEqual(
            get_total_instr_time_event(minimal_json, r".*PostgresEnv_shift_state$"),
            12.4918935,
        )
        self.assertAlmostEqual(
            get_total_instr_time_event(minimal_json, r".*Workload_execute$"),
            31.831543260000004,
        )
        self.assertAlmostEqual(
            get_total_instr_time_event(
                minimal_json, r".*(WolpPolicy_train_actor|WolpPolicy_train_critic)$"
            ),
            19.9834938712,
        )


if __name__ == "__main__":
    unittest.main()
