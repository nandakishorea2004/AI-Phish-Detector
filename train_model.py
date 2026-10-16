
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv("data/phishing_email.csv")

# 2. Prepare the data
df = df.dropna(subset=["text_combined", "label"])
df["text_combined"] = df["text_combined"].astype(str)

# Keep only the two expected classes
df = df[df["label"].isin([0, 1])]

X = df["text_combined"]
y = df["label"].astype(int)

print(f"Total emails: {len(df)}")
print("\nClass distribution:")
print(y.value_counts().sort_index())

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining emails: {len(X_train)}")
print(f"Testing emails: {len(X_test)}")

# 4. Build the machine-learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
            dtype=__import__("numpy").float32
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# 5. Train the model
print("\nTraining model. Please wait...")
model.fit(X_train, y_train)

# 6. Evaluate on unseen test data
print("\nEvaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Label 0", "Label 1"],
        digits=4,
        zero_division=0
    )
)

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# 7. Save the complete trained pipeline
joblib.dump(model, "phishing_model.joblib")

print("\nModel saved as phishing_model.joblib")
print("Training and evaluation completed.")
