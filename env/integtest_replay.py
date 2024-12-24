import unittest

from env.integtest_util import IntegtestWorkspace


class ReplayTests(unittest.TestCase):
    @staticmethod
    def setUpClass() -> None:
        IntegtestWorkspace.set_up_workspace()

    def test_replay(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
