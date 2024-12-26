import unittest

from env.tests.integtest_util import GymlibIntegtestManager
from env.tuning_artifacts import (
    DBMSConfigDelta,
    IndexesDelta,
    QueryKnobsDelta,
    SysKnobsDelta,
    TuningArtifactsReader,
    TuningArtifactsWriter,
)


class PostgresConnTests(unittest.TestCase):
    @staticmethod
    def setUpClass() -> None:
        GymlibIntegtestManager.set_up_workspace()

    @staticmethod
    def make_config(letter: str) -> DBMSConfigDelta:
        return DBMSConfigDelta(
            indexes=IndexesDelta([letter]),
            sysknobs=SysKnobsDelta({letter: letter}),
            qknobs=QueryKnobsDelta({letter: [letter]}),
        )

    def test_get_delta_at_step(self) -> None:
        writer = TuningArtifactsWriter(
            GymlibIntegtestManager.get_dbgym_workspace(),
            GymlibIntegtestManager.get_default_metadata(),
        )

        writer.write_step(PostgresConnTests.make_config("a"))
        writer.write_step(PostgresConnTests.make_config("b"))
        writer.write_step(PostgresConnTests.make_config("c"))

        reader = TuningArtifactsReader(writer.tuning_artifacts_dpath)

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
        writer = TuningArtifactsWriter(
            GymlibIntegtestManager.get_dbgym_workspace(),
            GymlibIntegtestManager.get_default_metadata(),
        )

        writer.write_step(PostgresConnTests.make_config("a"))
        writer.write_step(PostgresConnTests.make_config("b"))
        writer.write_step(PostgresConnTests.make_config("c"))

        reader = TuningArtifactsReader(writer.tuning_artifacts_dpath)

        self.assertEqual(
            reader.get_all_deltas_in_order(),
            [
                PostgresConnTests.make_config("a"),
                PostgresConnTests.make_config("b"),
                PostgresConnTests.make_config("c"),
            ],
        )

    def test_get_metadata(self) -> None:
        writer = TuningArtifactsWriter(
            GymlibIntegtestManager.get_dbgym_workspace(),
            GymlibIntegtestManager.get_default_metadata(),
        )
        reader = TuningArtifactsReader(writer.tuning_artifacts_dpath)
        metadata = reader.get_metadata()
        expected_metadata = GymlibIntegtestManager.get_default_metadata()
        self.assertEqual(metadata, expected_metadata)


if __name__ == "__main__":
    unittest.main()
