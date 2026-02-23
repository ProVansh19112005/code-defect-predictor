import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_commits.csv")

# Convert message to lowercase
df["message"] = df["message"].str.lower()

# Create bug_fix label
keywords = ["fix", "bug", "error", "patch", "issue"]

df["is_bug_fix"] = df["message"].apply(
    lambda x: 1 if any(word in x for word in keywords) else 0
)

print("Total rows:", len(df))
print("Bug-fix commits:", df["is_bug_fix"].sum())
print("Non-bug commits:", len(df) - df["is_bug_fix"].sum())

# Save labeled dataset
df.to_csv("data/labeled_commits.csv", index=False)

print("Labeling completed!")