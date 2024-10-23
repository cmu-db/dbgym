import os
import sys
import unittest

# Do this to suppress the logs we'd usually get when importing tensorflow
# By importing tensorflow in run_unit_tests.py, we avoid it being imported in any other file since run_unit_tests.py is always entered first when running unit tests.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow

del os.environ["TF_CPP_MIN_LOG_LEVEL"]

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(".")
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    if not result.wasSuccessful():
        # This is needed so that the GHA fails if the unit tests fail.
        sys.exit(1)
