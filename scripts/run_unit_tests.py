import os
import sys
import unittest

# See comment in the base task.py file for why we do this.
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
