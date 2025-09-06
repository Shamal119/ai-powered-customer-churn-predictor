import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

def clean_and_feature_engineer(df):
    """Clean data and create new features"""
    # Remove customerID as it's not useful for prediction
    df_clean = df.drop('customerID', axis=1)
    
    # Convert TotalCharges to numeric, replacing empty strings with NaN
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Handle missing values in TotalCharges (likely new customers with 0 tenure)
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
    
    # Convert SeniorCitizen to object type for consistency
    df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype(str)
    
    # Feature Engineering
    print("Creating new features...")
    
    # 1. Average monthly charges per tenure
    df_clean['AvgMonthlyCharges'] = df_clean['TotalCharges'] / (df_clean['tenure'] + 1)  # +1 to avoid division by zero
    
    # 2. Tenure groups
    df_clean['TenureGroup'] = pd.cut(df_clean['tenure'], 
                                    bins=[0, 12, 24, 48, 72], 
                                    labels=['New', 'Short', 'Medium', 'Long'])
    
    # 3. Monthly charges groups
    df_clean['MonthlyChargesGroup'] = pd.cut(df_clean['MonthlyCharges'], 
                                           bins=[0, 35, 70, 90, 120], 
                                           labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # 4. Total charges groups
    df_clean['TotalChargesGroup'] = pd.cut(df_clean['TotalCharges'], 
                                         bins=[0, 1000, 3000, 5000, 10000], 
                                         labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # 5. Service count (number of services)
    service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_clean['ServiceCount'] = 0
    for col in service_cols:
        df_clean['ServiceCount'] += (df_clean[col] != 'No').astype(int)
    
    # 6. Internet service type (binary)
    df_clean['HasInternet'] = (df_clean['InternetService'] != 'No').astype(int)
    
    # 7. Contract length (numeric)
    contract_mapping = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
    df_clean['ContractLength'] = df_clean['Contract'].map(contract_mapping)
    
    # 8. Payment method type
    df_clean['AutoPayment'] = df_clean['PaymentMethod'].str.contains('automatic').astype(int)
    
    print("Data after cleaning and feature engineering:")
    print(f"Shape: {df_clean.shape}")
    print(f"Missing values: {df_clean.isnull().sum().sum()}")
    
    return df_clean

def encode_categorical_features(df):
    """Encode categorical features"""
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
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

def train_and_evaluate_models(X, y):
    """Train and evaluate multiple models"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Handle class imbalance for XGBoost
        if name == 'XGBoost':
            model.set_params(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]))
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
    
    return results, scaler, X_train.columns.tolist(), X_test, y_test

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for the best model"""
    print("\nPerforming hyperparameter tuning for XGBoost...")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Create XGBoost model
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
    )
    
    # Grid search
    grid_search = GridSearchCV(
        xgb_model, 
        param_grid, 
        cv=5, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, scaler

def evaluate_final_model(model, scaler, X_test, y_test, feature_names):
    """Evaluate the final tuned model"""
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = (y_pred == y_test).mean()
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\nFinal Model Performance:")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return model, scaler, feature_names

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
    
    print("\nModel and preprocessor saved successfully!")

def create_visualizations(df, model, scaler, feature_names, X_test, y_test):
    """Create visualizations for the project"""
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Target distribution
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    df['Churn'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
    plt.title('Target Variable Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    
    # 2. Tenure distribution by churn
    plt.subplot(2, 3, 2)
    df.groupby('Churn')['tenure'].hist(alpha=0.7, bins=20)
    plt.title('Tenure Distribution by Churn')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Frequency')
    plt.legend(['No Churn', 'Churn'])
    
    # 3. Monthly charges by churn
    plt.subplot(2, 3, 3)
    df.groupby('Churn')['MonthlyCharges'].hist(alpha=0.7, bins=20)
    plt.title('Monthly Charges Distribution by Churn')
    plt.xlabel('Monthly Charges ($)')
    plt.ylabel('Frequency')
    plt.legend(['No Churn', 'Churn'])
    
    # 4. Contract type by churn
    plt.subplot(2, 3, 4)
    contract_churn = pd.crosstab(df['Contract'], df['Churn'])
    contract_churn.plot(kind='bar', stacked=True)
    plt.title('Contract Type by Churn')
    plt.xlabel('Contract Type')
    plt.ylabel('Count')
    plt.legend(['No Churn', 'Churn'])
    plt.xticks(rotation=45)
    
    # 5. Feature importance
    plt.subplot(2, 3, 5)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True).tail(10)
    
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Importance')
    
    # 6. ROC Curve
    plt.subplot(2, 3, 6)
    X_test_scaled = scaler.transform(X_test)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load and explore data
    df = load_and_explore_data()
    
    # Clean data and engineer features
    df_clean = clean_and_feature_engineer(df)
    
    # Encode categorical features
    df_encoded, label_encoders = encode_categorical_features(df_clean)
    
    # Prepare features and target
    X, y = prepare_features_and_target(df_encoded)
    
    # Train and evaluate multiple models
    results, scaler, feature_names, X_test, y_test = train_and_evaluate_models(X, y)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    print(f"\nBest model: {best_model_name}")
    print(f"Best ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    
    # Hyperparameter tuning for XGBoost
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    best_model, scaler = hyperparameter_tuning(X_train, y_train)
    
    # Evaluate final model
    final_model, scaler, feature_names = evaluate_final_model(
        best_model, scaler, X_test, y_test, feature_names
    )
    
    # Save model and preprocessor
    save_model_and_preprocessor(final_model, scaler, feature_names, label_encoders)
    
    # Create visualizations
    create_visualizations(df, final_model, scaler, feature_names, X_test, y_test)
