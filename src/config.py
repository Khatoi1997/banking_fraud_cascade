import os

# Data Layer Configuration
DATA_DIR = "data"
DATA_SOURCES = {
    "transactions": os.path.join(DATA_DIR, "Transaction_Master.csv"),
    "customers": os.path.join(DATA_DIR, "Customer_Profile.csv"),
    "accounts": os.path.join(DATA_DIR, "Account_Master.csv"),
    "logins": os.path.join(DATA_DIR, "Login_Activity.csv"),
    "devices": os.path.join(DATA_DIR, "Device_Inventory.csv"),
    "linked_accounts": os.path.join(DATA_DIR, "Linked_Accounts.csv"),
    "merchants": os.path.join(DATA_DIR, "Merchant_Blacklist.csv"),
    "alerts": os.path.join(DATA_DIR, "Alert_History_Labels.csv")
}

# Fraud Engine Parameters
VELOCITY_WINDOW_HOURS = 1
COST_FALSE_NEGATIVE = 5000  
COST_FALSE_POSITIVE = 50