import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Settings
start_date = datetime(2025, 1, 1)
end_date = datetime(2026, 5, 14)
num_customers = 1000
num_transactions = 10000

# 1. Customer Profile Generation
customers = pd.DataFrame({
    'customer_id': [f'CUST_{i}' for i in range(num_customers)],
    'age': np.random.randint(18, 85, num_customers),
    'kyc_status': np.random.choice(['Verified', 'Pending', 'Tier2'], num_customers),
    'home_lat': np.random.uniform(18.5, 19.5, num_customers),
    'home_lon': np.random.uniform(72.8, 73.0, num_customers)
})

# 2. Device Inventory Mapping (Establish Ground Truth Hardware)
devices = pd.DataFrame({
    'customer_id': customers['customer_id'],
    'trusted_device_id': [f'DEV_{i}' for i in range(num_customers)] # Linear clean assignment
})
trusted_map = devices.set_index('customer_id')['trusted_device_id'].to_dict()

# 3. Account Master Setup
accounts = pd.DataFrame({
    'account_id': [f'ACC_{i}' for i in range(num_customers)],
    'customer_id': customers['customer_id'],
    'account_balance': np.random.uniform(1000, 500000, num_customers),
    'date_opened': [start_date - timedelta(days=random.randint(1, 1000)) for _ in range(num_customers)]
})
acc_to_cust = accounts.set_index('account_id')['customer_id'].to_dict()

# Storage arrays for our behavioral cascades
tx_data = []
login_data = []
tx_counter = 0
login_counter = 0

print("Generating interlocked behavioral cascades...")

# 4. SIMULATION LOOP
while tx_counter < num_transactions:
    # Pick a random target account
    acc_idx = random.randint(0, num_customers - 1)
    acc_id = f'ACC_{acc_idx}'
    cust_id = acc_to_cust[acc_id]
    trusted_dev = trusted_map[cust_id]
    
    # Generate a random primary baseline event timestamp
    base_time = start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))
    
    # Decide if this specific interaction cluster is Fraud or Legitimate
    # A 0.08 cluster probability naturally yields roughly a 10% row-level fraud distribution 
    # because our fraud scenarios inject multiple rapid transactions back-to-back.
    is_fraud_cluster = random.random() < 0.08
    
    if not is_fraud_cluster:
        # ==========================================
        # GOOD USER BEHAVIOR
        # ==========================================
        # 1. Generates a standard baseline login from their verified hardware
        login_data.append([f'LOG_{login_counter}', cust_id, f'192.168.1.{random.randint(1,255)}', trusted_dev, base_time - timedelta(minutes=random.randint(2, 30))])
        login_counter += 1
        
        # 2. Executes a completely normal variable consumer transaction amount
        amount = np.random.exponential(scale=1500) + 10 # Naturally skewed realistic distribution
        amount = min(amount, 7000) # Cap normal spending
        tx_data.append([f'TX_{tx_counter}', acc_id, amount, base_time, random.choice(['Mobile', 'Web', 'ATM']), 0])
        tx_counter += 1
        
    else:
        # ==========================================
        # MALICIOUS BEHAVIOR CASCADE SCENARIOS
        # ==========================================
        scenario_choice = random.choice(['ATO', 'VELOCITY_BURST'])
        
        if scenario_choice == 'ATO':
            # SCENARIO 1: Account Takeover Execution
            # A thief logs in via unverified hardware, then bleeds the account out shortly after.
            untrusted_dev = f"DEV_{random.randint(2000, 5000)}" # Guaranteed non-matching device
            
            # Step A: Inject the malicious login footprint
            login_time = base_time - timedelta(minutes=random.randint(5, 120))
            login_data.append([f'LOG_{login_counter}', cust_id, '10.0.4.22', untrusted_dev, login_time])
            login_counter += 1
            
            # Step B: Inject a realistic micro-charge tester transaction
            tx_data.append([f'TX_{tx_counter}', acc_id, random.choice([0.45, 0.99, 1.00]), base_time, 'Web', 1])
            tx_counter += 1
            
            # Step C: Follow up with a massive balance extraction a few hours later
            strike_time = base_time + timedelta(hours=random.randint(1, 4))
            amount = np.random.uniform(8000, 14000) # Mixes right into normal high spend layers
            tx_data.append([f'TX_{tx_counter}', acc_id, amount, strike_time, 'Web', 1])
            tx_counter += 1
            
        elif scenario_choice == 'VELOCITY_BURST':
            # SCENARIO 5 & 6: Rapid Automated Velocity Overload
            # Attacker hits the account with multiple rapid transactions within a single hour
            login_data.append([f'LOG_{login_counter}', cust_id, '172.16.2.5', trusted_dev, base_time - timedelta(minutes=5)])
            login_counter += 1
            
            num_burst_txs = random.randint(3, 6)
            for step in range(num_burst_txs):
                burst_time = base_time + timedelta(minutes=step * random.randint(2, 10))
                # Distribute the fraud across variable amounts to prevent easy static matching
                amount = np.random.uniform(1500, 4500) 
                tx_data.append([f'TX_{tx_counter}', acc_id, amount, burst_time, 'Mobile', 1])
                tx_counter += 1

# Convert output buffers safely back to DataFrames
transactions = pd.DataFrame(tx_data[:num_transactions], columns=['tx_id', 'account_id', 'amount', 'timestamp', 'channel', 'is_fraud_label'])
logins = pd.DataFrame(login_data, columns=['login_id', 'customer_id', 'ip_address', 'device_id', 'timestamp'])

# Rest of your tables remain completely structurally preserved
alerts = transactions[transactions['is_fraud_label'] == 1][['tx_id', 'is_fraud_label']].copy()
alerts.columns = ['tx_id', 'investigation_result']

merchants = pd.DataFrame({'merchant_id': [f'MERCH_{i}' for i in range(100)], 'risk_score': np.random.uniform(0.1, 0.9, 100)})
linked = pd.DataFrame({'account_id_a': [f'ACC_{random.randint(0, 100)}' for _ in range(50)], 'account_id_b': [f'ACC_{random.randint(101, 200)}' for _ in range(50)], 'link_type': 'Shared_Phone'})

# Save to disk
customers.to_csv('Customer_Profile.csv', index=False)
accounts.to_csv('Account_Master.csv', index=False)
transactions.to_csv('Transaction_Master.csv', index=False)
logins.to_csv('Login_Activity.csv', index=False)
alerts.to_csv('Alert_History_Labels.csv', index=False)
merchants.to_csv('Merchant_Blacklist.csv', index=False)
linked.to_csv('Linked_Accounts.csv', index=False)
devices.to_csv('Device_Inventory.csv', index=False)

print(f"Ecosystem updated with complex behavioral loops! Total Transactions: {len(transactions)} | Total Logins: {len(logins)}")