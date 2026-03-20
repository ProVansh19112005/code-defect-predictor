import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("data/labeled_commits.csv")

# Features & target
X = df["message"]
y = df["is_bug_fix"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

X_vectorized = vectorizer.fit_transform(X)

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Save processed data
pickle.dump((X_vectorized, y), open("processed_data.pkl", "wb"))

print("Feature engineering completed!")