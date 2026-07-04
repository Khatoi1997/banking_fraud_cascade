import pandas as pd
from src.config import DATA_SOURCES

def load_and_consolidate_data():
    print("Inverting core banking CSVs into memory...")
    
    # 1. Load the Core Transaction Engine Files
    tx = pd.read_csv(DATA_SOURCES["transactions"])
    cust = pd.read_csv(DATA_SOURCES["customers"])
    acc = pd.read_csv(DATA_SOURCES["accounts"])
    tx['timestamp'] = pd.to_datetime(tx['timestamp'])
    
    # Consolidate baseline transaction data
    tx_enriched = tx.merge(acc[['account_id', 'customer_id', 'account_balance']], on='account_id', how='left')
    base_df = tx_enriched.merge(cust, on='customer_id', how='left')
    
    # 2. Load Identity & Access Logs (For Scenario 1: Account Takeover)
    logins = pd.read_csv(DATA_SOURCES["logins"])
    devices = pd.read_csv(DATA_SOURCES["devices"])
    logins['timestamp'] = pd.to_datetime(logins['timestamp'])
    
    # 3. Load Network/Mule Reference Files (For Network Scenarios)
    network = pd.read_csv(DATA_SOURCES["linked_accounts"])
    merchants = pd.read_csv(DATA_SOURCES["merchants"])
    alerts = pd.read_csv(DATA_SOURCES["alerts"])
    
    print(f"Core transaction engine ready. Shape: {base_df.shape}")
    
    # Return everything as a dictionary of domain data frames
    return {
        "transaction_base": base_df,
        "login_logs": logins,
        "device_inventory": devices,
        "network_links": network,
        "merchant_blacklist": merchants,
        "ground_truth_alerts": alerts
    }