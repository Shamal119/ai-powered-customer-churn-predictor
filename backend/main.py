from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict, Any
import os

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using XGBoost model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessor
model = None
scaler = None
feature_names = None
label_encoders = None

def load_model():
    """Load the trained model and preprocessor"""
    global model, scaler, feature_names, label_encoders
    
    try:
        # Load from the ml directory
        model = joblib.load('../ml/model.pkl')
        scaler = joblib.load('../ml/scaler.pkl')
        feature_names = joblib.load('../ml/feature_names.pkl')
        label_encoders = joblib.load('../ml/label_encoders.pkl')
        print("Model and preprocessor loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

# Pydantic model for input validation
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }

# Response model
class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float
    confidence: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Customer Churn Prediction API is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer_data: CustomerData):
    """
    Predict customer churn based on customer data
    """
    try:
        # Convert Pydantic model to dictionary
        data_dict = customer_data.dict()
        
        # Convert to DataFrame for preprocessing
        import pandas as pd
        df = pd.DataFrame([data_dict])
        
        # Apply the same preprocessing as training
        # Convert SeniorCitizen to string for consistency
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
        
        # Encode categorical features using the same label encoders
        df_encoded = df.copy()
        for col in label_encoders.keys():
            if col != 'Churn':  # Skip target variable
                le = label_encoders[col]
                # Handle unseen categories
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                df_encoded[col] = le.transform(df_encoded[col])
        
        # Select features in the same order as training
        X = df_encoded[feature_names]
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        prediction_proba = model.predict_proba(X_scaled)[0]
        
        # Convert prediction to human-readable format
        prediction_label = "Yes" if prediction == 1 else "No"
        churn_probability = float(prediction_proba[1])  # Probability of churn
        confidence = f"{churn_probability * 100:.1f}%"
        
        return PredictionResponse(
            prediction=prediction_label,
            churn_probability=churn_probability,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
