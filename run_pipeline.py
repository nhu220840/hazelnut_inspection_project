import sys
import time

try:
    from train import main as train_model
    from main import evaluate_system
except ImportError as e:
    print("❌ ERROR: Cannot find 'train.py' or 'main.py' file.")
    print(f"Details: {e}")
    sys.exit(1)


def run_full_pipeline():
    """Run training then evaluation; print timing and exit on error."""
    print("==================================================")
    print("🌰  HAZELNUT INSPECTION SYSTEM - AUTOMATED PIPELINE")
    print("==================================================")
    
    start_total = time.time()

    print("\n" + "="*30)
    print("🚀 [STEP 1/2] STARTING TRAINING PHASE...")
    print("="*30)
    start_train = time.time()
    
    try:
        train_model()
        print(f"✅ Training completed in {time.time() - start_train:.2f} seconds.")
    except Exception as e:
        print(f"❌ Training Failed! Error: {e}")
        sys.exit(1)

    print("\n" + "="*30)
    print("📊 [STEP 2/2] STARTING EVALUATION PHASE...")
    print("="*30)
    start_eval = time.time()
    
    try:
        evaluate_system()
        print(f"✅ Evaluation completed in {time.time() - start_eval:.2f} seconds.")
    except Exception as e:
        print(f"❌ Evaluation Failed! Error: {e}")
        sys.exit(1)

    total_duration = time.time() - start_total
    print("\n==================================================")
    print(f"✨ PIPELINE FINISHED SUCCESSFULLY!")
    print(f"⏱️  Total Execution Time: {total_duration:.2f} seconds")
    print("==================================================")

if __name__ == "__main__":
    run_full_pipeline()