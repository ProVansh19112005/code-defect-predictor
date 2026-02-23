import pandas as pd

# Load labeled dataset
df = pd.read_csv("data/labeled_commits.csv")

print("Loaded dataset shape:", df.shape)

# -------------------------------
# Basic Feature Engineering
# -------------------------------

df["code_churn"] = df["lines_added"] + df["lines_deleted"]
df["net_change"] = df["lines_added"] - df["lines_deleted"]

df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
df = df.dropna(subset=["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# -------------------------------
# Historical File-Level Features
# -------------------------------

# Sort by date to maintain history order
df = df.sort_values("date")

# Count previous modifications per file
df["previous_changes"] = df.groupby("file_name").cumcount()

# Count number of unique developers per file
developer_counts = df.groupby("file_name")["author"].nunique()
df["developer_count"] = df["file_name"].map(developer_counts)

# -------------------------------
# Final Feature Selection
# -------------------------------

final_df = df[[
    "lines_added",
    "lines_deleted",
    "nloc",
    "code_churn",
    "net_change",
    "previous_changes",
    "developer_count",
    "is_bug_fix"
]]

print("Final dataset shape:", final_df.shape)

final_df.to_csv("data/final_dataset.csv", index=False)

print("Feature engineering completed!")