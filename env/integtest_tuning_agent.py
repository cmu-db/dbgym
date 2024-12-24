import unittest

from env.integtest_util import IntegtestWorkspace, MockTuningAgent
from env.tuning_agent import (
    DBMSConfigDelta,
    IndexesDelta,
    QueryKnobsDelta,
    SysKnobsDelta,
    TuningAgentArtifactsReader,
)


class PostgresConnTests(unittest.TestCase):
    @staticmethod
    def setUpClass() -> None:
        IntegtestWorkspace.set_up_workspace()

    @staticmethod
    def make_config(letter: str) -> DBMSConfigDelta:
        return DBMSConfigDelta(
            indexes=IndexesDelta([letter]),
            sysknobs=SysKnobsDelta({letter: letter}),
            qknobs=QueryKnobsDelta({letter: [letter]}),
        )

    def test_get_delta_at_step(self) -> None:
        agent = MockTuningAgent(IntegtestWorkspace.get_dbgym_cfg())

        agent.config_to_return = PostgresConnTests.make_config("a")
        agent.step()
        agent.config_to_return = PostgresConnTests.make_config("b")
        agent.step()
        agent.config_to_return = PostgresConnTests.make_config("c")
        agent.step()

        reader = TuningAgentArtifactsReader(agent.tuning_agent_artifacts_dpath)

        self.assertEqual(
            reader.get_delta_at_step(1), PostgresConnTests.make_config("b")
        )
        self.assertEqual(
            reader.get_delta_at_step(0), PostgresConnTests.make_config("a")
        )
        self.assertEqual(
            reader.get_delta_at_step(1), PostgresConnTests.make_config("b")
        )
        self.assertEqual(
            reader.get_delta_at_step(2), PostgresConnTests.make_config("c")
        )

    def test_get_all_deltas_in_order(self) -> None:
        agent = MockTuningAgent(IntegtestWorkspace.get_dbgym_cfg())

        agent.config_to_return = PostgresConnTests.make_config("a")
        agent.step()
        agent.config_to_return = PostgresConnTests.make_config("b")
        agent.step()
        agent.config_to_return = PostgresConnTests.make_config("c")
        agent.step()

        reader = TuningAgentArtifactsReader(agent.tuning_agent_artifacts_dpath)

        self.assertEqual(
            reader.get_all_deltas_in_order(),
            [
                PostgresConnTests.make_config("a"),
                PostgresConnTests.make_config("b"),
                PostgresConnTests.make_config("c"),
            ],
        )

    def test_get_metadata(self) -> None:
        agent = MockTuningAgent(IntegtestWorkspace.get_dbgym_cfg())
        reader = TuningAgentArtifactsReader(agent.tuning_agent_artifacts_dpath)
        metadata = reader.get_metadata()
        expected_metadata = IntegtestWorkspace.get_default_metadata()
        self.assertEqual(metadata, expected_metadata)


if __name__ == "__main__":
    unittest.main()
