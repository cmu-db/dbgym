import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(".")
    print(f"suite={suite}")
    runner = unittest.TextTestRunner()
    runner.run(suite)
