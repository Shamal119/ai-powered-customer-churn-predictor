import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load and perform initial exploration of the dataset"""
    df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    print("Dataset Shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nTarget variable distribution:")
    print(df['Churn'].value_counts())
    print(f"Churn rate: {df['Churn'].value_counts()['Yes'] / len(df) * 100:.2f}%")
    
    return df

def clean_data(df):
    """Clean and preprocess the data"""
    # Remove customerID as it's not useful for prediction
    df_clean = df.drop('customerID', axis=1)
    
    # Convert TotalCharges to numeric, replacing empty strings with NaN
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Handle missing values in TotalCharges (likely new customers with 0 tenure)
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
    
    # Convert SeniorCitizen to object type for consistency
    df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype(str)
    
    print("Data after cleaning:")
    print(f"Shape: {df_clean.shape}")
    print(f"Missing values: {df_clean.isnull().sum().sum()}")
    
    return df_clean

def encode_categorical_features(df):
    """Encode categorical features"""
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns: {categorical_cols}")
    
    # Create label encoders for each categorical column
    label_encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
        print(f"{col}: {le.classes_}")
    
    return df_encoded, label_encoders

def prepare_features_and_target(df_encoded):
    """Prepare features and target variable"""
    # Separate features and target
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {y.value_counts()}")
    
    return X, y

def train_model(X, y):
    """Train XGBoost model"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])  # Handle class imbalance
    )
    
    xgb_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = xgb_model.predict(X_test_scaled)
    y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate model
    print("Model Performance:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    return xgb_model, scaler, X_train.columns.tolist()

def save_model_and_preprocessor(model, scaler, feature_names, label_encoders):
    """Save the trained model and preprocessor"""
    # Save model
    joblib.dump(model, 'model.pkl')
    
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save feature names
    joblib.dump(feature_names, 'feature_names.pkl')
    
    # Save label encoders
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    print("Model and preprocessor saved successfully!")

if __name__ == "__main__":
    # Load and explore data
    df = load_and_explore_data()
    
    # Clean data
    df_clean = clean_data(df)
    
    # Encode categorical features
    df_encoded, label_encoders = encode_categorical_features(df_clean)
    
    # Prepare features and target
    X, y = prepare_features_and_target(df_encoded)
    
    # Train model
    model, scaler, feature_names = train_model(X, y)
    
    # Save model and preprocessor
    save_model_and_preprocessor(model, scaler, feature_names, label_encoders)
