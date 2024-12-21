import subprocess
import unittest
from typing import Any, Optional

from env.integtest_util import (
    INTEGTEST_DBGYM_CFG,
    get_integtest_workspace_path,
    set_up_integtest_workspace,
)
from env.tuning_agent import DBMSConfigDelta, TuningAgent


class MockTuningAgent(TuningAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config_to_return: Optional[DBMSConfigDelta] = None

    def _step(self) -> DBMSConfigDelta:
        assert self.config_to_return is not None
        ret = self.config_to_return
        # Setting this ensures you must set self.config_to_return every time.
        self.config_to_return = None
        return ret


class PostgresConnTests(unittest.TestCase):
    @staticmethod
    def setUpClass() -> None:
        set_up_integtest_workspace()

    @staticmethod
    def make_config(letter: str) -> DBMSConfigDelta:
        return DBMSConfigDelta([letter], {letter: letter}, {letter: [letter]})

    def test_get_step_delta(self) -> None:
        agent = MockTuningAgent(INTEGTEST_DBGYM_CFG)

        agent.config_to_return = PostgresConnTests.make_config("a")
        agent.step()
        agent.config_to_return = PostgresConnTests.make_config("b")
        agent.step()
        agent.config_to_return = PostgresConnTests.make_config("c")
        agent.step()

        self.assertEqual(agent.get_step_delta(1), PostgresConnTests.make_config("b"))
        self.assertEqual(agent.get_step_delta(0), PostgresConnTests.make_config("a"))
        self.assertEqual(agent.get_step_delta(1), PostgresConnTests.make_config("b"))
        self.assertEqual(agent.get_step_delta(2), PostgresConnTests.make_config("c"))

    def test_get_all_deltas(self) -> None:
        agent = MockTuningAgent(INTEGTEST_DBGYM_CFG)

        agent.config_to_return = PostgresConnTests.make_config("a")
        agent.step()
        agent.config_to_return = PostgresConnTests.make_config("b")
        agent.step()
        agent.config_to_return = PostgresConnTests.make_config("c")
        agent.step()

        self.assertEqual(
            agent.get_all_deltas(),
            [
                PostgresConnTests.make_config("a"),
                PostgresConnTests.make_config("b"),
                PostgresConnTests.make_config("c"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
