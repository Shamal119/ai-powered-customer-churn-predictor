#!/bin/bash

echo "🎨 Starting Frontend Development Server"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "vercel.json" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

echo "📦 Installing frontend dependencies..."
cd frontend
npm install

echo ""
echo "🚀 Starting React development server..."
echo "Frontend will be available at: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the frontend server"
npm start
