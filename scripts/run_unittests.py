import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='test*.py')
    runner = unittest.TextTestRunner()
    runner.run(suite)