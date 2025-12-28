"""Machine learning model training pipeline."""

import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)


def train():
    """
    Train RandomForest classifier for churn prediction.
    
    Loads data, splits into train/test sets, trains pipeline,
    evaluates metrics, and saves pipeline and metrics to disk.
    """
    # Get absolute path to data file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "data.csv")

    # Load customer data
    df = pd.read_csv(DATA_PATH)

    # Define features and target
    X_cols = ["age", "tenure_months", "monthly_charges", "contract_type", "support_tickets"]
    X = df[X_cols]
    y = df["churn"]

    # Split data (80/20 with fixed random state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create preprocessing and model pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier())
    ])

    # Train pipeline
    pipeline.fit(X_train, y_train)

    # Save trained pipeline
    joblib.dump(pipeline, "../pipeline.pkl")

    # Generate predictions and calculate metrics
    y_pred = pipeline.predict(X_test)

    metrics = {
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 2),
        "Precision": round(float(precision_score(y_test, y_pred)), 2),
        "Recall": round(float(recall_score(y_test, y_pred)), 2),
        "F1 Score": round(float(f1_score(y_test, y_pred)), 2)
    }

    # Save metrics
    joblib.dump(metrics, "../metrics.pkl")

    # Print results
    print(metrics)
    print(classification_report(y_test, y_pred))