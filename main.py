from fastapi import FastAPI
import joblib

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0"
)

model = joblib.load(
    "model/heart_disease_extra_trees.pkl"
)


@app.get("/")
def root():
    return {
        "status": "success",
        "message": "Heart Disease API is running"
    }
}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }