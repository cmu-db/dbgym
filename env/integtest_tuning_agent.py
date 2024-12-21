import subprocess
import unittest

from env.integtest_util import (
    ENV_INTEGTESTS_DBGYM_CONFIG_FPATH,
    get_integtest_workspace_path,
)
from env.tuning_agent import DBMSConfig, TuningAgent
from util.workspace import DBGymConfig


class MockTuningAgent(TuningAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config_to_return = None

    def _step(self) -> DBMSConfig:
        assert self.config_to_return is not None
        ret = self.config_to_return
        # Setting this ensures you must set self.config_to_return every time.
        self.config_to_return = None
        return ret


class PostgresConnTests(unittest.TestCase):
    dbgym_cfg: DBGymConfig

    @staticmethod
    def setUpClass() -> None:
        # If you're running the test locally, this check makes runs past the first one much faster.
        if not get_integtest_workspace_path().exists():
            subprocess.run(["./env/set_up_env_integtests.sh"], check=True)

        PostgresConnTests.dbgym_cfg = DBGymConfig(ENV_INTEGTESTS_DBGYM_CONFIG_FPATH)

    @staticmethod
    def make_config(letter: str) -> DBMSConfig:
        return DBMSConfig([letter], {letter: letter}, {letter: [letter]})

    def test_get_step_delta(self) -> None:
        agent = MockTuningAgent(PostgresConnTests.dbgym_cfg)

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
        agent = MockTuningAgent(PostgresConnTests.dbgym_cfg)

        agent.config_to_return = PostgresConnTests.make_config("a")
        agent.step()
        agent.config_to_return = PostgresConnTests.make_config("b")
        agent.step()
        agent.config_to_return = PostgresConnTests.make_config("c")
        agent.step()

        self.assertEqual(agent.get_all_deltas(), [PostgresConnTests.make_config("a"), PostgresConnTests.make_config("b"), PostgresConnTests.make_config("c")])


if __name__ == "__main__":
    unittest.main()
