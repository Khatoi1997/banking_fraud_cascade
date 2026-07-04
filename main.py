import sys
import os

# Align workspace directories
sys.path.append(os.getcwd())

from src.train import run_production_training_pipeline
from src.evaluation import run_model_evaluation_suite

def run_master_operational_pipeline():
    print("==================================================================")
    print("🛰️  LAUNCHING FULL ENTERPRISE FRAUD CASCADE OPERATIONS PIPELINE")
    print("==================================================================")
    
    # Phase 1: Ingest, Engineer Features, Train, and Serialize the Model Artifact
    try:
        run_production_training_pipeline()
    except Exception as e:
        print(f"❌ CRITICAL PIPELINE FAILURE DURING TRAINING PHASE: {str(e)}")
        sys.exit(1)
        
    print("\n" + "="*50 + "\n")
    
    # Phase 2: Run the Independent Compliance and Financial Risk Audit
    try:
        run_model_evaluation_suite()
    except Exception as e:
        print(f"❌ CRITICAL PIPELINE FAILURE DURING COMPLIANCE EVALUATION PHASE: {str(e)}")
        sys.exit(1)

    print("\n==================================================================")
    print("✅ MASTER PIPELINE RUN COMPLETE: Model Trained, Verified, and Exported")
    print("==================================================================")

if __name__ == "__main__":
    run_master_operational_pipeline()