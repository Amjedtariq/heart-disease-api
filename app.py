from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "model/heart_disease_extra_trees.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("AI model loaded successfully.")
except Exception as e:
    model = None
    print("Error loading model:", e)


# ============================================================
# REQUEST MODEL
# ============================================================

class PatientData(BaseModel):

    Age: float
    Gender: str

    Weight: float
    Height: float
    BMI: float

    Smoking: str
    Alcohol_Intake: str
    Physical_Activity: str
    Diet: str
    Stress_Level: str

    Diabetes: str
    Hyperlipidemia: str
    Family_History: str
    Previous_Heart_Attack: str

    Systolic_BP: float
    Diastolic_BP: float
    Heart_Rate: float

    Blood_Sugar_Fasting: float
    Cholesterol_Total: float

    Hypertension: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "success",
        "message": "Heart Disease AI API is running"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(data: PatientData):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="AI model is not loaded"
        )

    try:

        # Convert request to dictionary
        patient_dict = data.model_dump()

        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_dict])

        # Prediction
        prediction = model.predict(patient_df)[0]

        # Probability
        probabilities = model.predict_proba(patient_df)[0]

        probability = float(probabilities[1]) * 100

        return {

            "success": True,

            "prediction": int(prediction),

            "probability": round(probability, 2),

            "message":
                "Heart disease detected"
                if int(prediction) == 1
                else
                "No heart disease detected"

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )