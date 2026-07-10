import sys
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append("src")

from machineguard.features import add_engineered_features
from machineguard.modeling import ENGINEERED_FEATURE_COLS

MODEL_PATH = Path("artifacts/model.joblib")
THRESHOLD=0.3

app = FastAPI(title="MachineGuard AI")
model = joblib.load(MODEL_PATH)

class MachineInput(BaseModel):
    type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_failure(machine:MachineInput):
    machine_data = {
        "Type": machine.type,
        "Air temperature [K]": machine.air_temperature,
        "Process temperature [K]": machine.process_temperature,
        "Rotational speed [rpm]": machine.rotational_speed,
        "Torque [Nm]": machine.torque,
        "Tool wear [min]": machine.tool_wear
    } 
    machine_df = pd.DataFrame([machine_data])
    machine_df = add_engineered_features(machine_df)
    X = machine_df[ENGINEERED_FEATURE_COLS]

    failure_prob = model.predict_proba(X)[:,1][0]
    predicted_failure = int(failure_prob >=THRESHOLD)

    return{
        "failure_probability": round(float(failure_prob),4),
        "predicted_failure": predicted_failure,
        "threshold":THRESHOLD
    }