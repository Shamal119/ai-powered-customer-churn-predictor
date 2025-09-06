# 🚀 Local Setup Guide

## Quick Start (Recommended)

### Option 1: Using the provided scripts

**Terminal 1 - Backend:**
```bash
./run_local.sh
```

**Terminal 2 - Frontend:**
```bash
./run_frontend.sh
```

### Option 2: Manual setup

## Step-by-Step Manual Setup

### Prerequisites
- Python 3.9+
- Node.js (v14 or higher)
- npm or yarn

### Step 1: Train the Machine Learning Model
```bash
cd ml
pip install -r requirements.txt
python improved_model.py
```
*This may take 5-10 minutes due to hyperparameter tuning*

### Step 2: Start the Backend API
```bash
cd ../backend
pip install -r requirements.txt
python main.py
```
*Backend will be available at: http://localhost:8000*

### Step 3: Start the Frontend (in a new terminal)
```bash
cd frontend
npm install
npm start
```
*Frontend will be available at: http://localhost:3000*

## 🌐 Access Points

Once everything is running:

- **Frontend Application**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## 🧪 Testing the API

You can test the API directly:
```bash
python test_api.py
```

## 📱 Using the Application

1. **Open your browser** and go to http://localhost:3000
2. **Try the Predictor**: Fill out the customer form and get real-time predictions
3. **Explore Project Details**: Click "Project Details" to see the ML showcase
4. **View API Docs**: Go to http://localhost:8000/docs for interactive API documentation

## 🔧 Troubleshooting

### Common Issues

1. **Port already in use**:
   - Backend: Change port in `backend/main.py` (line 133)
   - Frontend: React will ask to use a different port

2. **Model not found**:
   - Make sure you ran `python improved_model.py` in the `ml` directory first

3. **Dependencies not installed**:
   - Backend: `pip install -r backend/requirements.txt`
   - Frontend: `npm install` in the frontend directory

4. **CORS errors**:
   - Make sure the backend is running on port 8000
   - Check that CORS is enabled in the backend

### Performance Notes

- **Model Training**: First run takes 5-10 minutes due to hyperparameter tuning
- **API Response**: Should be under 500ms for predictions
- **Frontend Load**: Should load in under 3 seconds

## 🎯 What You'll See

### Frontend Features
- **Interactive Predictor**: Form with all customer features
- **Real-time Predictions**: Instant results with confidence scores
- **Project Showcase**: Detailed ML project documentation
- **Responsive Design**: Works on desktop, tablet, and mobile

### Backend Features
- **RESTful API**: Clean API endpoints
- **Auto Documentation**: Swagger UI at /docs
- **Input Validation**: Pydantic models for data validation
- **Error Handling**: Comprehensive error responses

## 🚀 Next Steps

1. **Try the Predictor**: Test with different customer scenarios
2. **Explore the Code**: Check out the ML pipeline and frontend components
3. **Deploy to Vercel**: Use `vercel --prod` to deploy to production
4. **Customize**: Modify the model or add new features

## 📞 Support

If you encounter any issues:
1. Check the console for error messages
2. Verify all dependencies are installed
3. Make sure both backend and frontend are running
4. Check the API documentation at http://localhost:8000/docs

---

**🎉 Enjoy exploring your AI-Powered Customer Churn Predictor!**
