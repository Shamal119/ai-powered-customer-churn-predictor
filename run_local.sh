#!/bin/bash

echo "🚀 Starting AI-Powered Customer Churn Predictor Locally"
echo "======================================================"

# Check if we're in the right directory
if [ ! -f "vercel.json" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

echo "📊 Step 1: Training the ML model..."
cd ml
if [ ! -f "model.pkl" ]; then
    echo "Training model (this may take a few minutes)..."
    python improved_model.py
    if [ $? -eq 0 ]; then
        echo "✅ Model trained successfully!"
    else
        echo "❌ Model training failed"
        exit 1
    fi
else
    echo "✅ Model already exists, skipping training"
fi

echo ""
echo "🔧 Step 2: Installing backend dependencies..."
cd ../backend
pip install -r requirements.txt

echo ""
echo "🌐 Step 3: Starting backend server..."
echo "Backend will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the backend server"
python main.py
