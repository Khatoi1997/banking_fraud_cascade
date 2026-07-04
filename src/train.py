import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

# Align workspace directories
sys.path.append(os.getcwd())

from src.data_loader import load_and_consolidate_data
from src.features import engineer_fraud_features
from src.models import get_xgboost_classifier

def run_production_training_pipeline():
    print("==================================================")
    print("Starting Automated Production Model Training Loop")
    print("==================================================")
    
    # 1. Ingest multi-domain raw data tables
    data_blocks = load_and_consolidate_data()
    
    # 2. Execute cross-domain feature engineering pipeline
    processed_df = engineer_fraud_features(data_blocks)
    
    # 3. Sanitize types and quiet Pandas downcast warnings
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    processed_df[numeric_cols] = processed_df[numeric_cols].fillna(0)
    
    categorical_cols = ['channel', 'kyc_status']
    for col in categorical_cols:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype('category').cat.codes

    # 4. Enforce strict Chronological Out-of-Time Validation Split
    cutoff_date = pd.to_datetime('2026-02-06')
    train_mask = processed_df['timestamp'] < cutoff_date
    val_mask = processed_df['timestamp'] >= cutoff_date
    
    train_df = processed_df[train_mask]
    val_df = processed_df[val_mask]
    
    # Isolate targets and clear identifiers to protect from data leakage
    drop_features = ['tx_id', 'account_id', 'customer_id', 'timestamp', 'actual_login_time', 'is_fraud_label']
    
    X_train = train_df.drop(columns=drop_features)
    y_train = train_df['is_fraud_label']
    X_val = val_df.drop(columns=drop_features)
    y_val = val_df['is_fraud_label']
    
    print(f"\nTraining set size: {X_train.shape[0]} rows | Validation set size: {X_val.shape[0]} rows")
    
    # Calculate scale balance weight directly from training distributions
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    calculated_weight = neg_count / pos_count if pos_count > 0 else 4.0
    
    # 5. Load model architecture from models.py factory registry
    print("\nLoading model engine from src.models registry...")
    model = get_xgboost_classifier(scale_pos_weight=calculated_weight)
    
    # 6. Fit the production model
    print("Training production algorithm on historical horizon...")
    model.fit(X_train, y_train)
    
    # 7. Evaluate on unseen OOT data to verify safety margins
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    print("\n=== Validation Performance Report ===")
    print(classification_report(y_val, y_pred, target_names=['Legitimate', 'Fraudulent']))
    print(f"OOT ROC-AUC Score: {roc_auc_score(y_val, y_proba):.4f}")
    
    # 8. Serialize and export the model artifact to disk
    os.makedirs('models', exist_ok=True)
    artifact_path = os.path.join('models', 'xgboost_fraud_cascade_v1.pkl')
    
    with open(artifact_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"\nSuccess! Production model serialized to disk at: {artifact_path}")
    print("==================================================")

if __name__ == "__main__":
    run_production_training_pipeline()