import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Align workspace paths
sys.path.append(os.getcwd())

from src.data_loader import load_and_consolidate_data
from src.features import engineer_fraud_features

def run_model_evaluation_suite():
    print("==================================================")
    print("Initializing Independent Model Evaluation Suite")
    print("==================================================")
    
    # 1. Paths verification
    model_path = os.path.join('models', 'xgboost_fraud_cascade_v1.pkl')
    if not os.path.exists(model_path):
        print(f"CRITICAL ERROR: Saved model binary not found at '{model_path}'. Please run src/train.py first.")
        return

    # 2. Re-ingest and process data block ecosystem for validation slice
    data_blocks = load_and_consolidate_data()
    processed_df = engineer_fraud_features(data_blocks)
    
    # 3. Apply identical variable type sanitization
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(0)
    
    categorical_cols = ['channel', 'kyc_status']
    for col in categorical_cols:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype('category').cat.codes

    # 4. Extract strict Out-of-Time Validation slice (Matching our training baseline)
    cutoff_date = pd.to_datetime('2026-02-06')
    val_df = processed_df[processed_df['timestamp'] >= cutoff_date]
    
    drop_features = ['tx_id', 'account_id', 'customer_id', 'timestamp', 'actual_login_time', 'is_fraud_label']
    X_val = val_df.drop(columns=drop_features)
    y_val = val_df['is_fraud_label']
    
    # 5. Load model artifact directly from memory disk
    print(f"\nLoading saved model artifact from: {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # 6. Execute model inference predictions
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # 7. Compute Standard Performance Metrics
    print("\n--- 📊 Statistical Performance Matrix ---")
    print(classification_report(y_val, y_pred, target_names=['Legitimate', 'Fraudulent']))
    print(f"ROC-AUC Performance Score: {roc_auc_score(y_val, y_proba):.4f}")
    
    # 8. Compute Financial Confusion Matrix Breakdown
    print("\n--- 💼 Financial Risk Operations Audit ---")
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    
    # Extract financial amount values to calculate monetary impact
    val_amounts = val_df['amount'].values
    
    total_fraud_attempted_value = val_amounts[y_val == 1].sum()
    fraud_prevented_value = val_amounts[(y_val == 1) & (y_pred == 1)].sum()
    fraud_leakage_value = val_amounts[(y_val == 1) & (y_pred == 0)].sum()
    false_alarm_user_friction_value = val_amounts[(y_val == 0) & (y_pred == 1)].sum()
    
    print(f"True Negatives  (Legitimate Allowed Cleared) : {tn:4d} accounts | Financial Volume: ${val_amounts[(y_val == 0) & (y_pred == 0)].sum():,.2f}")
    print(f"False Positives (False Alarms / User Friction): {fp:4d} accounts | Financial Volume: ${false_alarm_user_friction_value:,.2f}")
    print(f"False Negatives (Missed Fraud / Cash Leakage) : {fn:4d} accounts | Financial Volume: ${fraud_leakage_value:,.2f}")
    print(f"True Positives  (Fraud Stopped Securely)     : {tp:4d} accounts | Financial Volume: ${fraud_prevented_value:,.2f}")
    print("-" * 50)
    print(f"Total Fraud Attack Volume Encountered        : ${total_fraud_attempted_value:,.2f}")
    print(f"Total Fraud Cash Successfully Saved         : ${fraud_prevented_value:,.2f}")
    print(f"Total Fraud Leakage Loss                     : ${fraud_leakage_value:,.2f}")
    
    if total_fraud_attempted_value > 0:
        mitigation_rate = (fraud_prevented_value / total_fraud_attempted_value) * 100
        print(f"Financial Mitigation Efficiency Rate         : {mitigation_rate:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    run_model_evaluation_suite()

    