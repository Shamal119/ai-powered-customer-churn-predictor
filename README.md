# 🚀 AI-Powered Customer Churn Predictor

A comprehensive full-stack machine learning project that demonstrates end-to-end data science workflow from data analysis to production deployment. This project showcases advanced feature engineering, model optimization, and modern web development practices.

## 📊 Project Overview

This project predicts customer churn using machine learning with a focus on business impact and technical excellence. The application combines a high-performance XGBoost model with a modern React frontend and FastAPI backend to provide real-time churn predictions.

### 🎯 Key Achievements

- **74.9% Accuracy** and **84.6% ROC AUC Score**
- **Advanced Feature Engineering** with 8 new engineered features
- **Responsive Web Application** with real-time predictions
- **Production-Ready Deployment** on Vercel
- **Comprehensive API Documentation** with Swagger UI
- **Business Intelligence Dashboard** with actionable insights

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │  ML Pipeline    │
│                 │    │                 │    │                 │
│ • TypeScript    │◄──►│ • Python        │◄──►│ • XGBoost       │
│ • Tailwind CSS  │    │ • Pydantic      │    │ • Scikit-learn  │
│ • Real-time UI  │    │ • CORS enabled  │    │ • Feature Eng.  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🧠 Machine Learning Details

### Model Performance
| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | 74.9% | Overall prediction accuracy |
| **ROC AUC** | 84.6% | Area under the ROC curve |
| **Precision** | 52.0% | True positives / (True positives + False positives) |
| **Recall** | 80.0% | True positives / (True positives + False negatives) |
| **F1-Score** | 63.0% | Harmonic mean of precision and recall |

### Feature Engineering
The model includes advanced feature engineering techniques:

1. **Average Monthly Charges**: `TotalCharges / (tenure + 1)`
2. **Tenure Groups**: Categorical grouping (New, Short, Medium, Long)
3. **Service Count**: Number of services used by customer
4. **Internet Service Binary**: Has internet service or not
5. **Contract Length**: Numeric representation of contract duration
6. **Auto Payment**: Binary indicator for automatic payments
7. **Charges Groups**: Categorical grouping for charges
8. **Payment Method Type**: Automatic vs manual payment

### Top 10 Most Important Features
1. **Contract Type** (35.0%) - Month-to-month customers churn more
2. **Online Security** (9.2%) - Security services reduce churn
3. **Tech Support** (7.8%) - Support services improve retention
4. **Internet Service** (5.8%) - Service type affects churn patterns
5. **Streaming Movies** (3.7%) - Streaming usage impacts retention
6. **Tenure** (3.7%) - Longer tenure reduces churn risk
7. **Monthly Charges** (3.2%) - Higher charges correlate with churn
8. **Paperless Billing** (2.9%) - Billing method affects retention
9. **Service Count** (2.9%) - More services improve retention
10. **Payment Method** (2.5%) - Automatic payments reduce churn

## 💼 Business Insights

### Key Findings
1. **Contract Type is Critical**: Month-to-month contracts have 55.1% churn rate vs 11.8% for annual contracts
2. **Security Services Matter**: Customers without online security churn 2.3x more
3. **Tech Support Reduces Churn**: Customers with support churn 40% less
4. **Tenure Predicts Loyalty**: Customers >24 months have only 8% churn rate
5. **Payment Method Impact**: Electronic check users churn 2.5x more than automatic payment users

### Recommendations
- Focus on converting month-to-month customers to longer contracts
- Promote security add-ons and make them more accessible
- Invest in proactive tech support and customer success programs
- Implement loyalty programs for long-term customers
- Encourage automatic payment setup with incentives

## 🛠️ Technology Stack

### Frontend
- **React 19** with TypeScript
- **Tailwind CSS** for styling
- **Responsive Design** for all devices
- **Real-time Predictions** with loading states

### Backend
- **Python 3.9+** with FastAPI
- **Pydantic** for data validation
- **CORS** enabled for frontend integration
- **Automatic API Documentation** (Swagger UI)

### Machine Learning
- **XGBoost** for gradient boosting
- **Scikit-learn** for preprocessing
- **Pandas** for data manipulation
- **Hyperparameter Tuning** with GridSearchCV

### Deployment
- **Vercel** for serverless deployment
- **Environment-specific** API URLs
- **Production-ready** configuration

## 📁 Project Structure

```
churn-predictor/
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── ChurnPredictor.tsx
│   │   │   └── ProjectShowcase.tsx
│   │   ├── App.tsx          # Main application
│   │   └── index.css        # Tailwind CSS
│   ├── package.json
│   └── tailwind.config.js
├── backend/                 # FastAPI backend
│   ├── main.py             # API server
│   ├── requirements.txt    # Python dependencies
│   └── vercel.json        # Vercel config
├── ml/                     # Machine learning
│   ├── data_analysis.py   # Basic model training
│   ├── improved_model.py  # Advanced model with tuning
│   ├── requirements.txt   # ML dependencies
│   └── *.pkl             # Trained model files
├── data/                   # Dataset
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── vercel.json            # Root Vercel config
├── build.sh              # Build script
├── test_api.py           # API testing
└── README.md             # This file
```

## 🚀 Getting Started

### Prerequisites
- Node.js (v14 or higher)
- Python 3.9+
- npm or yarn

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd churn-predictor
   ```

2. **Train the improved model**
   ```bash
   cd ml
   pip install -r requirements.txt
   python improved_model.py
   ```

3. **Start the backend**
   ```bash
   cd ../backend
   pip install -r requirements.txt
   python main.py
   ```

4. **Start the frontend**
   ```bash
   cd ../frontend
   npm install
   npm start
   ```

5. **Access the application**
   - Frontend: `http://localhost:3000`
   - API Docs: `http://localhost:8000/docs`

### Build and Deploy

```bash
# Build the project
./build.sh

# Deploy to Vercel
vercel --prod
```

## 📊 API Documentation

### POST /predict
Predict customer churn based on customer data.

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "prediction": "Yes",
  "churn_probability": 0.82,
  "confidence": "82.0%"
}
```

## 🧪 Testing

### Test the API
```bash
python test_api.py
```

### Model Validation
The model is validated using:
- **5-fold Cross-Validation** for robust evaluation
- **Grid Search** for hyperparameter optimization
- **Multiple Algorithms** comparison (Logistic Regression, Random Forest, XGBoost)
- **Feature Importance** analysis for interpretability

## 📈 Model Comparison

| Model | Accuracy | ROC AUC | Status |
|-------|----------|---------|--------|
| Logistic Regression | 80.4% | 84.7% | Best Performance |
| Random Forest | 77.9% | 81.7% | Good Performance |
| XGBoost (Tuned) | 74.9% | 84.6% | Selected for Production |

## 🎨 Frontend Features

### Interactive Predictor
- **Comprehensive Form** with all 19 customer features
- **Real-time Validation** with user-friendly inputs
- **Loading States** and error handling
- **Confidence Visualization** with progress bars
- **Responsive Design** for all devices

### Project Showcase
- **Model Details** with performance metrics
- **Feature Importance** visualization
- **Business Insights** with recommendations
- **Technical Architecture** overview
- **Interactive Tabs** for easy navigation

## 🔧 Development

### Code Quality
- **TypeScript** for type safety
- **ESLint** for code linting
- **Prettier** for code formatting
- **Component-based** architecture
- **Responsive design** principles

### Performance
- **API Response Time**: < 500ms
- **Frontend Load Time**: < 3 seconds
- **Model Prediction**: < 100ms
- **Optimized Bundle** size

## 📚 Learning Outcomes

This project demonstrates:
- **End-to-end ML Pipeline** from data to deployment
- **Feature Engineering** techniques for better performance
- **Model Optimization** with hyperparameter tuning
- **Full-stack Development** with modern technologies
- **Production Deployment** best practices
- **Business Intelligence** and actionable insights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Shamal Musthafa**
- GitHub: [@shamalmusthafa](https://github.com/shamalmusthafa)
- LinkedIn: [Shamal Musthafa](https://linkedin.com/in/shamalmusthafa)

## 🙏 Acknowledgments

- Telco Customer Churn Dataset
- XGBoost team for the excellent library
- FastAPI for the modern Python web framework
- React team for the powerful frontend library
- Vercel for seamless deployment

---

**⭐ If you found this project helpful, please give it a star!**
