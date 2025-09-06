#!/usr/bin/env python3
"""
Test script for the Customer Churn Prediction API
"""

import requests
import json
import time

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    # Test data
    test_data = {
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
    
    print("Testing Customer Churn Prediction API...")
    print("=" * 50)
    
    # Test health endpoint
    try:
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the backend is running on port 8000")
        return
    
    # Test prediction endpoint
    try:
        print("\n2. Testing prediction endpoint...")
        response = requests.post(
            f"{base_url}/predict",
            headers={"Content-Type": "application/json"},
            json=test_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Churn Probability: {result['churn_probability']:.3f}")
            print(f"   Confidence: {result['confidence']}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")
    
    print("\n" + "=" * 50)
    print("API testing completed!")

if __name__ == "__main__":
    test_api()
