import os
import sys
import pickle
import pandas as pd
import numpy as np

# Align workspace paths
sys.path.append(os.getcwd())

class RealTimeInferenceEngine:
    def __init__(self):
        """
        Initializes the inference worker by loading the serialized model artifact into cache memory.
        """
        self.model_path = os.path.join('models', 'xgboost_fraud_cascade_v1.pkl')
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Production model binary missing at {self.model_path}. Run training first.")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        # Extract the exact feature names the model was trained on
        self.model_features = self.model.get_booster().feature_names
        print("Real-Time Inference Engine successfully spun up and cached.")

    def evaluate_single_transaction(self, current_tx, historical_context_blocks):
        """
        Accepts a single incoming transaction event dict along with its corresponding 
        historical data domain context blocks to calculate an immediate risk score.
        """
        # 1. Convert incoming transaction dictionary into a single-row DataFrame
        incoming_df = pd.DataFrame([current_tx])
        incoming_df['timestamp'] = pd.to_datetime(incoming_df['timestamp'])
        
        # 2. Extract operational logs from data blocks
        tx_base = historical_context_blocks["transaction_base"].copy()
        logins = historical_context_blocks["login_logs"].copy()
        devices = historical_context_blocks["device_inventory"].copy()
        
        # Append the new transaction to historical transaction records to allow rolling window calculations
        tx_base = pd.concat([tx_base, incoming_df], ignore_index=True)
        tx_base = tx_base.sort_values(by=['account_id', 'timestamp']).reset_index(drop=True)
        
        # 3. Calculate 1-Hour Rolling Windows on the fly for this account
        account_group = tx_base[tx_base['account_id'] == current_tx['account_id']].copy()
        account_group.set_index('timestamp', inplace=True)
        
        # Pull rolling counts and sums specifically up to the exact execution time of the current transaction
        rolling_window = account_group.rolling('1h')
        account_group['tx_count_last_1h'] = rolling_window['amount'].count()
        account_group['amt_spent_last_1h'] = rolling_window['amount'].sum()
        
        # Extract the computed rolling stats back out for just our incoming row
        target_timestamp = pd.to_datetime(current_tx['timestamp'])
        computed_row = account_group.loc[[target_timestamp]].tail(1).reset_index()
        
        # 4. Process Scenario 9: Test Transaction Detection
        previous_txs = tx_base[tx_base['account_id'] == current_tx['account_id']]
        if len(previous_txs) > 1:
            prev_amt = previous_txs.iloc[-2]['amount']
            is_test_pattern = 1 if (prev_amt < 1.0 and current_tx['amount'] > 100) else 0
        else:
            is_test_pattern = 0
        computed_row['is_test_tx_pattern'] = is_test_pattern

        # 5. Process Cross-Domain Identity Matching (Scenario 1: ATO Risk)
        logins = logins.sort_values(by='timestamp').reset_index(drop=True)
        trusted_map = devices.set_index('customer_id')['trusted_device_id'].to_dict()
        logins['trusted_device_id'] = logins['customer_id'].map(trusted_map)
        logins['is_untrusted_login'] = (logins['device_id'] != logins['trusted_device_id']).astype(int)
        logins['actual_login_time'] = logins['timestamp']
        
        # Align incoming record for global merge_asof comparison
        computed_row = computed_row.sort_values(by='timestamp').reset_index(drop=True)
        
        enriched_row = pd.merge_asof(
            computed_row,
            logins[['customer_id', 'timestamp', 'actual_login_time', 'is_untrusted_login']],
            on='timestamp',
            by='customer_id',
            direction='nearest',
            tolerance=pd.Timedelta('24h')
        )
        
        # Calculate session delta time gap
        enriched_row['login_tx_time_diff_hours'] = (
            (enriched_row['timestamp'] - enriched_row['actual_login_time']).dt.total_seconds().abs() / 3600.0
        )
        enriched_row['is_ato_risk_24h'] = np.where((enriched_row['is_untrusted_login'] == 1), 1, 0)
        
        # 6. Sanitize missing links & encode categories to match model training exactly
        categorical_mapping = {
            'channel': {'Mobile': 0, 'Web': 1, 'ATM': 2},
            'kyc_status': {'Verified': 0, 'Pending': 1, 'Tier2': 2}
        }
        for col, mapping in categorical_mapping.items():
            if col in enriched_row.columns:
                val = enriched_row.loc[0, col]
                enriched_row[col] = mapping.get(val, 0)
                
        numeric_cols = enriched_row.select_dtypes(include=[np.number]).columns
        enriched_row[numeric_cols] = enriched_row[numeric_cols].fillna(0)
        
        # 7. Align columns perfectly with training feature list
        X_live = enriched_row[self.model_features]
        
        # 8. Score transaction via model inference probability weights
        risk_probability = float(self.model.predict_proba(X_live)[:, 1][0])
        final_prediction = int(self.model.predict(X_live)[0])
        
        return {
            "risk_probability": risk_probability,
            "action_recommendation": "DECLINE" if risk_probability > 0.50 else "APPROVE",
            "computed_features": X_live.to_dict(orient='records')[0]
        }