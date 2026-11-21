
import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")

# Display dataset
print("Dataset:")
print(df.head(10))

# Check dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check label distribution
print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nTotal Records:", len(df))

