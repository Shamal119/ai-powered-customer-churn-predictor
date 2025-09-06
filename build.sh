#!/bin/bash

echo "Building AI-Powered Customer Churn Predictor..."
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "vercel.json" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Build frontend
echo "1. Building frontend..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi
npm run build
cd ..

# Copy model files to backend for deployment
echo "2. Preparing model files..."
cp ml/*.pkl backend/

# Create a simple test
echo "3. Testing API..."
cd backend
python -c "
import joblib
import os
print('Model files check:')
for file in ['model.pkl', 'scaler.pkl', 'feature_names.pkl', 'label_encoders.pkl']:
    if os.path.exists(file):
        print(f'✅ {file}')
    else:
        print(f'❌ {file} missing')
"
cd ..

echo "4. Build completed!"
echo "You can now deploy to Vercel with: vercel --prod"
