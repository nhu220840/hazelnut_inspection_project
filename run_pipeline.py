import sys
import time
import os

# Import main functions from 2 files train.py and main.py
# Note: Python will automatically find train.py and main.py in the same directory
try:
    from train import main as train_model
    from main import evaluate_system
except ImportError as e:
    print("❌ ERROR: Cannot find 'train.py' or 'main.py' file.")
    print(f"Details: {e}")
    sys.exit(1)

def run_full_pipeline():
    """
    Function to execute the entire End-to-End process
    """
    print("==================================================")
    print("🌰  HAZELNUT INSPECTION SYSTEM - AUTOMATED PIPELINE")
    print("==================================================")
    
    start_total = time.time()

    # --- STEP 1: TRAINING ---
    print("\n" + "="*30)
    print("🚀 [STEP 1/2] STARTING TRAINING PHASE...")
    print("="*30)
    start_train = time.time()
    
    try:
        # Call main() function inside train.py
        train_model()
        print(f"✅ Training completed in {time.time() - start_train:.2f} seconds.")
    except Exception as e:
        print(f"❌ Training Failed! Error: {e}")
        # If training fails, stop immediately, don't evaluate
        sys.exit(1)

    # --- STEP 2: EVALUATION ---
    print("\n" + "="*30)
    print("📊 [STEP 2/2] STARTING EVALUATION PHASE...")
    print("="*30)
    start_eval = time.time()
    
    try:
        # Call evaluate_system() function inside main.py
        evaluate_system()
        print(f"✅ Evaluation completed in {time.time() - start_eval:.2f} seconds.")
    except Exception as e:
        print(f"❌ Evaluation Failed! Error: {e}")
        sys.exit(1)

    # --- SUMMARY ---
    total_duration = time.time() - start_total
    print("\n==================================================")
    print(f"✨ PIPELINE FINISHED SUCCESSFULLY!")
    print(f"⏱️  Total Execution Time: {total_duration:.2f} seconds")
    print("==================================================")

if __name__ == "__main__":
    run_full_pipeline()