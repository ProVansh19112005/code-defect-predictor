import pandas as pd

# Load dataset
df = pd.read_csv("data/commits.csv")

print("Original shape:", df.shape)

# Keep only Python files
df = df[df["file_name"].str.endswith(".py", na=False)]

# Remove missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

print("After cleaning shape:", df.shape)

# Save cleaned version
df.to_csv("data/cleaned_commits.csv", index=False)

print("Cleaning completed!")