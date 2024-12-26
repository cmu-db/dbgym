import os
import sys
import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(".", pattern=sys.argv[1])
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    if not result.wasSuccessful():
        # This is needed so that the GHA fails if the unit tests fail.
        sys.exit(1)
