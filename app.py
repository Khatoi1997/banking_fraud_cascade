import streamlit as st
import pandas as pd
from datetime import datetime
import time
import os
import sys

# Align workspace paths
sys.path.append(os.getcwd())

from src.data_loader import load_and_consolidate_data
from src.inference import RealTimeInferenceEngine

# Set up page configurations
st.set_page_config(
    page_title="Cascade Risk Engine Room",
    page_icon="🛡️",
    layout="wide"
)

# 1. Cache the underlying database tables to keep UI blazing fast
@st.cache_data
def get_cached_banking_context():
    return load_and_consolidate_data()

# Initialize Engine and Core Context Data
if "inference_engine" not in st.session_state:
    st.session_state["inference_engine"] = RealTimeInferenceEngine()
    st.session_state["context_blocks"] = get_cached_banking_context()

engine = st.session_state["inference_engine"]
context = st.session_state["context_blocks"]


# --- FAIL-SAFE ACCOUNT DATA DETECTION ---
# Automatically find whichever dictionary key contains our account data frame
account_key = None
for key in context.keys():
    if isinstance(context[key], pd.DataFrame) and 'account_id' in context[key].columns:
        account_key = key
        break

if account_key is None:
    st.error("🚨 CRITICAL ERROR: Could not find any data table containing an 'account_id' column inside your data loader!")
    st.stop()

# Safely extract UNIQUE profiles using the dynamically discovered key
available_accounts = context[account_key]["account_id"].unique()[:20].tolist()
account_to_customer_map = context[account_key].set_index('account_id')['customer_id'].to_dict()
# ----------------------------------------


# --- HEADER REGION ---
st.title("🛡️ Enterprise Cross-Domain Fraud Cascade Architecture")
st.markdown("""
This production dashboard runs live real-time inference using the trained **XGBoost Cascade Model** artifact.
Toggle individual behavioral triggers, transactional velocities, and security contexts below to witness how the model responds.
""")
st.write("---")

# --- USER INPUT COLUMNS ---
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📥 Transaction Payload Ingestion")
    
    # Core Ingestion Forms
    selected_acc = st.selectbox("Select Target Bank Account:", available_accounts)
    target_cust = account_to_customer_map[selected_acc]
    
    amount = st.number_input("Transaction Amount ($):", min_value=0.0, max_value=50000.0, value=1250.0, step=50.0)
    channel = st.selectbox("Transaction Channel:", ["Mobile", "Web", "ATM"])
    
    st.subheader("🌐 Session Identity Context")
    is_untrusted = st.checkbox("Simulate Login from Untrusted Device / Suspicious IP", value=False)
    
    st.subheader("⚡ Rolling Velocity Controls")
    velocity_override = st.slider(
        "Force Simulated Transaction Count (Past 1 Hour):",
        min_value=1, max_value=10, value=1,
        help="Simulates an active burst velocity pattern on the target account."
    )

with col2:
    st.header("⚡ Live Core-Banking Evaluation Engine")
    
    # 2. Package current UI configurations into a mock single transaction stream event
    mock_transaction = {
        "tx_id": "TX_LIVE_SIM_999",
        "account_id": selected_acc,
        "customer_id": target_cust,
        "amount": amount,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "channel": channel,
        "kyc_status": "Verified" # Default mapping criteria
    }
    
    # 3. Inject explicit UI session contextual configurations over raw historical state
    if is_untrusted:
        # Override the most recent login block row for this user to trigger untrusted status
        context["login_logs"].loc[context["login_logs"]['customer_id'] == target_cust, 'device_id'] = "DEV_UNKNOWN_ATTACKER_999"
        context["login_logs"].loc[context["login_logs"]['customer_id'] == target_cust, 'timestamp'] = pd.Timestamp.now() - pd.Timedelta(minutes=15)
    else:
        # Align with their verified hardware profile mapping criteria
        trusted_dev_id = context["device_inventory"].set_index('customer_id')['trusted_device_id'].to_dict().get(target_cust, "DEV_0")
        context["login_logs"].loc[context["login_logs"]['customer_id'] == target_cust, 'device_id'] = trusted_dev_id
        context["login_logs"].loc[context["login_logs"]['customer_id'] == target_cust, 'timestamp'] = pd.Timestamp.now() - pd.Timedelta(minutes=30)

    # Trigger Engine Pipeline Execution
    if st.button("🚀 Process & Audit Incoming Transaction", use_container_width=True):
        
        with st.spinner("Executing cross-domain feature windowing matrices..."):
            # Execute pipeline
            result = engine.evaluate_single_transaction(mock_transaction, context)
            
            # Manual injection override to reflect user's slider input choice
            result["computed_features"]["tx_count_last_1h"] = velocity_override
            
            # Re-predict probability to match manual velocity updates cleanly
            features_df = pd.DataFrame([result["computed_features"]])
            risk_prob_updated = float(engine.model.predict_proba(features_df)[:, 1][0])
            action = "DECLINE" if risk_prob_updated > 0.50 else "APPROVE"

        # Display Final Operational System Verdict
        if action == "DECLINE":
            st.error(f"🛑 CRITICAL VERDICT: {action} (Risk Probability Score: {risk_prob_updated:.4%})")
        else:
            st.success(f"✅ APPROVED: Transaction Cleared Safely (Risk Probability Score: {risk_prob_updated:.4%})")
            
        st.write("---")
        
        # Breakdown Feature Interpretations Grid Dashboard
        st.subheader("🕵️‍♂️ Real-Time Feature Matrix Explainer")
        
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        
        with feat_col1:
            st.metric(
                label="Untrusted Hardware Footprint",
                value="MALICIOUS FOOTPRINT" if result["computed_features"]["is_untrusted_login"] == 1 else "TRUSTED HARDWARE",
                delta="Risk High" if result["computed_features"]["is_untrusted_login"] == 1 else "Safe Baseline",
                delta_color="inverse"
            )
            
        with feat_col2:
            st.metric(
                label="Hourly Account Transaction Velocity",
                value=f"{int(velocity_override)} txs / hr",
                delta="Velocity Surge" if velocity_override >= 3 else "Normal Activity",
                delta_color="inverse" if velocity_override >= 3 else "normal"
            )
            
        with feat_col3:
            st.metric(
                label="Login-to-Transaction Time Gap",
                value=f"{result['computed_features']['login_tx_time_diff_hours']:.2f} Hours",
                help="Time elapsed between device authentication event and payment clearance event."
            )
            
        # Raw Data Stream View Mode for Audit compliance logs
        with st.expander("📄 View Enriched Core Ingestion JSON Log Payload"):
            st.json({
                "raw_event_payload": mock_transaction,
                "engineered_features_passed_to_model": result["computed_features"]
            })

        # --- AUTOMATED TERMINAL SECURITY CLOSURE TIMEOUT ---
        st.write("---")
        
        # Create a placeholder in the UI for the countdown message
        countdown_placeholder = st.empty()
        
        # Execute the 10-second countdown loop
        for seconds_left in range(10, 0, -1):
            countdown_placeholder.warning(f"🔒 SECURITY TIMEOUT: Stopping local web server and killing VS Code terminal session in {seconds_left} seconds...")
            time.sleep(1)
            
        # Display final exit warning
        countdown_placeholder.error("💥 Killing active terminal thread. Connection lost.")
        time.sleep(0.5)
        
        # Shutdown the Streamlit web server and kill the parent backend Python process completely
        os._exit(0)