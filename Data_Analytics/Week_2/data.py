

import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

# First 5 rows
print("First 5 rows:")
print(df.head())

# Column names
print("\nColumn Names:")
print(df.columns)

# Data types
print("\nData Types:")
print(df.dtypes)

# Shape of dataset
print("\nDataset Size:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

