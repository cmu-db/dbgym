import logging
import sys
from pathlib import Path

import pandas as pd


def read_and_output_parquet(file_path: Path) -> None:
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(file_path)

    # Output the DataFrame
    logging.getLogger("output").info("DataFrame:")
    logging.getLogger("output").info(df)


if __name__ == "__main__":
    # Specify the path to the Parquet file
    parquet_file_path = Path(sys.argv[0])

    # Call the function to read and output the Parquet file
    read_and_output_parquet(parquet_file_path)
