from pydriller import Repository
import pandas as pd

repo_url = "https://github.com/pallets/flask"

data = []
commit_count = 0

print("Mining repository...")

for commit in Repository(repo_url, only_in_branch="main").traverse_commits():
    commit_count += 1
    print(f"Processing commit #{commit_count}: {commit.hash}")

    for modification in commit.modified_files:
        data.append({
            "commit_hash": commit.hash,
            "author": commit.author.name,
            "date": commit.author_date,
            "message": commit.msg,
            "file_name": modification.filename,
            "lines_added": modification.added_lines,
            "lines_deleted": modification.deleted_lines,
            "nloc": modification.nloc
        })

print("Finished mining. Saving CSV...")

df = pd.DataFrame(data)
df.to_csv("data/commits.csv", index=False)

print("Data extraction completed!")
print("Total commits processed:", commit_count)
print("Total rows:", len(df))