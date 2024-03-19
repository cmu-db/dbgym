import sys

import pandas as pd


def read_and_print_parquet(file_path):
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(file_path)

    # Print the DataFrame
    print("DataFrame:")
    print(df)


if __name__ == "__main__":
    # Specify the path to the Parquet file
    parquet_file_path = sys.argv[0]

    # Call the function to read and print the Parquet file
    read_and_print_parquet(parquet_file_path)
