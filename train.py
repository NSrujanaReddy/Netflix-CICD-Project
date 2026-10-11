
import pandas as pd
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Netflix dataset
DATA_PATH = "NetflixDataset.csv"

df = pd.read_csv(DATA_PATH)

print("Netflix Dataset Loaded Successfully")
print("Dataset shape:", df.shape)
print(df.head())

# Select required columns
required_columns = ["type", "rating", "country", "duration", "listed_in"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

# Prepare the data
df = df[required_columns].fillna("Unknown")
df["type"] = df["type"].str.strip()

# Create input text and target labels
X = (
    "Rating: " + df["rating"].astype(str)
    + " Country: " + df["country"].astype(str)
    + " Duration: " + df["duration"].astype(str)
    + " Genre: " + df["listed_in"].astype(str)
)

y = df["type"]

# Encode target labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Create and train model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nModel Training Completed")
print("Accuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=list(range(len(encoder.classes_))),
    target_names=encoder.classes_,
    zero_division=0
))

# Save model and encoder
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/netflix_model.joblib")
joblib.dump(encoder, "artifacts/label_encoder.joblib")

# Save metrics for the CI/CD pipeline
with open("artifacts/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f, indent=4)

print("\nModel and metrics saved in artifacts folder")
