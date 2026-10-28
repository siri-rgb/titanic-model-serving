from typing import Optional, Literal
import joblib, numpy as np, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Titanic Survival API")
model = joblib.load("model.pkl")

class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: Literal["male", "female"]
    Age: Optional[float] = Field(None, ge=0, le=120)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    Embarked: Optional[Literal["S", "C", "Q"]] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(p: Passenger):
    data = {k: (np.nan if v is None else v) for k, v in p.model_dump().items()}
    df = pd.DataFrame([data])
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {"prediction": pred,
            "label": "Survived" if pred == 1 else "Did not survive",
            "survival_probability": round(prob, 4)}
