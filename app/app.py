"""
FastAPI application for customer churn prediction.

This module provides REST API endpoints for:
- Health monitoring
- Model metrics retrieval
- Model retraining
- Churn prediction on new data
"""

from fastapi import FastAPI
from app.schemas import ChurnInput
import pandas as pd
import joblib
from training.training_model import train


app = FastAPI()

# Load pre-trained machine learning pipeline and metrics
model = joblib.load("pipeline.pkl")
metrics = joblib.load("metrics.pkl")


@app.get("/")
def health_check():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        dict: Status message indicating API health
    """
    return {"status": "API is running"}


@app.get("/model/metrics")
def metrics_check():
    """
    Retrieve model performance metrics.
    
    Returns:
        dict: Model evaluation metrics (accuracy, precision, recall, etc.)
    """
    return metrics


@app.post("/model/retrain")
def retrain_model():
    """
    Retrain the machine learning model with updated data.
    
    Returns:
        dict: Status of retraining operation and updated metrics
    """
    train()
    return {
        "Status": "Trained",
        "Metrics": metrics
    }


@app.post("/model/predict")
def predict_churn(data: ChurnInput):
    """
    Predict customer churn probability for given input features.
    
    Args:
        data (ChurnInput): Input features for churn prediction
        
    Returns:
        dict: Churn prediction label and probability percentage
    """
    # Convert input data to DataFrame for model compatibility
    df = pd.DataFrame([data.dict()])
    
    # Generate prediction and probability from the pipeline
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    # Map numeric prediction to human-readable label
    if int(prediction) == 0:
        prediction = "Not Churn"
    elif int(prediction) == 1:
        prediction = "Churn"
    else:
        prediction = "Error"

    return {
        "churn_prediction": prediction,
        "churn_probability": round(float(probability), 3) * 100
    }