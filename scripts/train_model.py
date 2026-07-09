import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.append("src")

from machineguard.modeling import (
    TARGET_COL,
    FEATURE_COLS,
    build_random_forest_model,
)

DATA_PATH = Path("data/ai4i2020.csv")
THRESHOLD = 0.3

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

df = pd.read_csv(DATA_PATH)

X = df[FEATURE_COLS]
y = df[TARGET_COL]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full,
)

model = build_random_forest_model()
model.fit(X_train, y_train)

test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= THRESHOLD).astype(int)

metrics = calculate_metrics(y_test, test_predictions)
matrix = confusion_matrix(y_test, test_predictions)

print("Test metrics:")
print(metrics)

print("\nConfusion matrix:")
print(matrix)