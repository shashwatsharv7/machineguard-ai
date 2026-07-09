import sys
from pathlib import Path
import joblib
import pandas as pd
sys.path.append("src")
from machineguard.features import add_engineered_features
from machineguard.modeling import ENGINEERED_FEATURE_COLS

MODEL_PATH = Path("artifacts/model.joblib")
THRESHOLD = 0.3

machine = {
    "Type": "L",
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 0,
}

model = joblib.load(MODEL_PATH)

machine_df = pd.DataFrame([machine])
machine_df = add_engineered_features(machine_df)

X = machine_df[ENGINEERED_FEATURE_COLS]

failure_prob = model.predict_proba(X)[:,1][0]
predicted_failure = int(failure_prob >= THRESHOLD)

print("Failure Probability:", round(failure_prob,4))
print("Predicted failure:", predicted_failure)
print("Threshold:", THRESHOLD)
