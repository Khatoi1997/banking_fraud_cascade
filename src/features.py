import pandas as pd
import numpy as np

def engineer_fraud_features(data_blocks):
    """
    Engineers behavioral features for the 15 fraud scenarios across multiple data domains.
    Accepts a dictionary containing all 8 data tables.
    """
    print("Engineering features for the 15 fraud scenarios across data domains...")
    
    # Extract operational tables cleanly from the dictionary payload
    df = data_blocks["transaction_base"].copy()
    logins = data_blocks["login_logs"].copy()
    devices = data_blocks["device_inventory"].copy()
    
    # =========================================================================
    # 1. TRANSACTION LAYER FEATURES (Scenarios 5, 6, & 9)
    # =========================================================================
    # Ensure correct chronological order within each account
    df = df.sort_values(by=['account_id', 'timestamp']).reset_index(drop=True)
    
    # Robust isolated window function to bypass duplicate timestamp index tracking errors
    def calculate_rolling_metrics(group):
        # Set timestamp as a localized temporary index for time-windowing
        temp_df = pd.DataFrame({
            'count_counter': 1,
            'amount_tracker': group['amount']
        }, index=group['timestamp'])
        
        rolling_window = temp_df.rolling('1h')
        
        # Pull values out as clean independent lists
        counts = rolling_window['count_counter'].count().values
        sums = rolling_window['amount_tracker'].sum().values
        return list(zip(counts, sums))

    # Apply rolling metrics per account safely
    print("Calculating rolling transactional velocity and structuring flags...")
    rolling_outputs = (
        df.groupby('account_id', group_keys=False)
        .apply(calculate_rolling_metrics, include_groups=False)
        .explode()
        .values
    )
    
    # Split the exploded tuples cleanly back into separate feature targets
    df['tx_count_last_1h'] = [item[0] for item in rolling_outputs]
    df['amt_spent_last_1h'] = [item[1] for item in rolling_outputs]
    
    # Enforce explicit data types
    df['tx_count_last_1h'] = df['tx_count_last_1h'].astype(int)
    df['amt_spent_last_1h'] = df['amt_spent_last_1h'].astype(float)
    
    # Scenario 9: Test Transactions (Find if previous transaction was a micro-charge)
    df['prev_amt'] = df.groupby('account_id')['amount'].shift(1)
    df['is_test_tx_pattern'] = ((df['prev_amt'] < 1.0) & (df['amount'] > 100)).astype(int)
    
    # Drop intermediate column to avoid predictive leakage or noise
    df.drop(columns=['prev_amt'], inplace=True)
    
    # =========================================================================
    # 2. CROSS-DOMAIN IDENTITY LAYER FEATURES (Scenario 1: Account Takeover)
    # =========================================================================
    print("Cross-referencing device profiles and login sessions...")
    logins = logins.sort_values(by='timestamp').reset_index(drop=True)
    
    # Build a lookup map of customer to their verified trusted hardware ID
    trusted_map = devices.set_index('customer_id')['trusted_device_id'].to_dict()
    logins['trusted_device_id'] = logins['customer_id'].map(trusted_map)
    
    # Flag logins originating from unverified hardware
    logins['is_untrusted_login'] = (logins['device_id'] != logins['trusted_device_id']).astype(int)
    
    # Create a shadow duplicate column of the timestamp to survive pd.merge_asof
    logins['actual_login_time'] = logins['timestamp']
    
    # Sort the core transaction engine globally by timestamp (strict merge_asof constraint)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Link every transaction back to its NEAREST login session window
    df_enriched = pd.merge_asof(
        df,
        logins[['customer_id', 'timestamp', 'actual_login_time', 'is_untrusted_login']],
        on='timestamp',
        by='customer_id',
        direction='nearest',
        tolerance=pd.Timedelta('24h')
    )
    
    # Calculate exact absolute session time difference in hours
    df_enriched['login_tx_time_diff_hours'] = (
        (df_enriched['timestamp'] - df_enriched['actual_login_time']).dt.total_seconds().abs() / 3600.0
    )
    
    # Scenario 1: Establish Final Binary Cascade Risk parameter
    df_enriched['is_ato_risk_24h'] = np.where(
        (df_enriched['is_untrusted_login'] == 1), 
        1, 
        0
    )
    
    # =========================================================================
    # 3. PIPELINE SANITIZATION
    # =========================================================================
    # Clean up empty spaces left behind from customers without overlapping logs
    df_enriched['is_untrusted_login'] = df_enriched['is_untrusted_login'].fillna(0).astype(int)
    df_enriched['is_ato_risk_24h'] = df_enriched['is_ato_risk_24h'].fillna(0).astype(int)
    df_enriched['login_tx_time_diff_hours'] = df_enriched['login_tx_time_diff_hours'].fillna(9999.0)
    
    # Final global safety fill for any remaining structural NaN mutations
    df_enriched.fillna(0, inplace=True)
    
    print(f"Feature engineering successful. Total Enriched Frame Shape: {df_enriched.shape}")
    return df_enriched