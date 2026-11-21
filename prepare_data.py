
import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset.csv")

# Remove missing values and duplicates
df = df.dropna()
df = df.drop_duplicates()

# Convert data into instruction format
data = []

for _, row in df.iterrows():

    sample = {
        "instruction": "Classify the sentiment of this customer feedback as Positive, Negative, or Neutral.",
        "input": row["text"],
        "output": row["label"]
    }

    data.append(sample)

# Split dataset
train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Save training dataset
with open("train.json", "w") as f:
    json.dump(train_data, f, indent=4)

# Save testing dataset
with open("test.json", "w") as f:
    json.dump(test_data, f, indent=4)

print("Dataset preparation completed!")
print("Training records:", len(train_data))
print("Testing records:", len(test_data))
