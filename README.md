# Customer Churn Prediction API

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Details](#model-details)
- [Configuration](#configuration)

## Features

- **Health Monitoring**: Check API status with health check endpoint
- **Churn Prediction**: Predict customer churn probability with confidence scores
- **Model Metrics**: Retrieve model performance metrics (accuracy, precision, recall, F1-score)
- **Model Retraining**: Retrain model with updated data on demand
- **Automatic Validation**: Pydantic-based request validation with automatic OpenAPI documentation

## Project Structure

```
Project-01/
├── app/
│   ├── app.py              # FastAPI application and endpoints
│   └── schemas.py          # Pydantic data models for validation
├── training/
│   ├── training_model.py   # ML training pipeline
│   └── __init__.py
├── data/
│   └── data.csv            # Customer dataset
├── pipeline.pkl            # Trained model pipeline (generated)
├── metrics.pkl             # Model metrics (generated)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd /media/areeb/Data/Workspace/ML+FastAPI/Project-01
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if pipeline.pkl doesn't exist)
   ```bash
   python -c "from training.training_model import train; train()"
   ```

5. **Start the API server**
   ```bash
   uvicorn app.app:app --reload
   ```

   The API will be available at `http://localhost:8000`

## Usage

### Interactive API Documentation

Once the server is running, access the interactive API docs:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Example Requests

#### Health Check
```bash
curl -X GET "http://localhost:8000/"
```

**Response:**
```json
{
  "status": "API is running"
}
```

#### Get Model Metrics
```bash
curl -X GET "http://localhost:8000/model/metrics"
```

**Response:**
```json
{
  "Accuracy": 0.85,
  "Precision": 0.82,
  "Recall": 0.88,
  "F1 Score": 0.85
}
```

#### Predict Churn
```bash
curl -X POST "http://localhost:8000/model/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "tenure_months": 24,
    "monthly_charges": 89.95,
    "contract_type": 1,
    "support_tickets": 3
  }'
```

**Response:**
```json
{
  "churn_prediction": "Churn",
  "churn_probability": 87.5
}
```

#### Retrain Model
```bash
curl -X POST "http://localhost:8000/model/retrain"
```

**Response:**
```json
{
  "Status": "Trained",
  "Metrics": {
    "Accuracy": 0.86,
    "Precision": 0.83,
    "Recall": 0.89,
    "F1 Score": 0.86
  }
}
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check endpoint |
| GET | `/model/metrics` | Retrieve model performance metrics |
| POST | `/model/predict` | Predict churn for customer |
| POST | `/model/retrain` | Retrain model with updated data |

### Request/Response Schemas

#### ChurnInput (POST /model/predict)

```json
{
  "age": integer,
  "tenure_months": integer,
  "monthly_charges": float,
  "contract_type": integer (0=Month-to-month, 1=One year, 2=Two year),
  "support_tickets": integer
}
```

#### Prediction Response

```json
{
  "churn_prediction": string ("Churn", "Not Churn", or "Error"),
  "churn_probability": float (0-100)
}
```

## Model Details

### Algorithm
- **Model Type**: RandomForestClassifier
- **Preprocessing**: StandardScaler (feature normalization)
- **Training/Test Split**: 80/20

### Features Used
1. **age**: Customer age in years
2. **tenure_months**: Customer relationship duration in months
3. **monthly_charges**: Monthly subscription/service cost
4. **contract_type**: Type of contract (encoded as integer)
5. **support_tickets**: Number of support tickets filed

### Target Variable
- **churn**: Binary classification (0 = No churn, 1 = Churn)

### Evaluation Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Configuration

### Data Location
Update the data path in `training/training_model.py` if needed:
```python
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "data.csv")
```

### Model Artifacts
- `pipeline.pkl`: Serialized trained model (generated after training)
- `metrics.pkl`: Model evaluation metrics (generated after training)

### Dependencies
See `pyproject.toml` for complete list of packages.

---

**Last Updated**: December 29, 2025