import sys
import time
import os

# Import các hàm chính từ 2 file train.py và main.py
# Lưu ý: Python sẽ tự tìm file train.py và main.py trong cùng thư mục
try:
    from train import main as train_model
    from main import evaluate_system
except ImportError as e:
    print("❌ LỖI: Không tìm thấy file 'train.py' hoặc 'main.py'.")
    print(f"Chi tiết: {e}")
    sys.exit(1)

def run_full_pipeline():
    """
    Hàm thực thi toàn bộ quy trình End-to-End
    """
    print("==================================================")
    print("🌰  HAZELNUT INSPECTION SYSTEM - AUTOMATED PIPELINE")
    print("==================================================")
    
    start_total = time.time()

    # --- BƯỚC 1: HUẤN LUYỆN (TRAINING) ---
    print("\n" + "="*30)
    print("🚀 [STEP 1/2] STARTING TRAINING PHASE...")
    print("="*30)
    start_train = time.time()
    
    try:
        # Gọi hàm main() bên trong train.py
        train_model()
        print(f"✅ Training completed in {time.time() - start_train:.2f} seconds.")
    except Exception as e:
        print(f"❌ Training Failed! Error: {e}")
        # Nếu train lỗi thì dừng ngay, không đánh giá nữa
        sys.exit(1)

    # --- BƯỚC 2: ĐÁNH GIÁ (EVALUATION) ---
    print("\n" + "="*30)
    print("📊 [STEP 2/2] STARTING EVALUATION PHASE...")
    print("="*30)
    start_eval = time.time()
    
    try:
        # Gọi hàm evaluate_system() bên trong main.py
        evaluate_system()
        print(f"✅ Evaluation completed in {time.time() - start_eval:.2f} seconds.")
    except Exception as e:
        print(f"❌ Evaluation Failed! Error: {e}")
        sys.exit(1)

    # --- TỔNG KẾT ---
    total_duration = time.time() - start_total
    print("\n==================================================")
    print(f"✨ PIPELINE FINISHED SUCCESSFULLY!")
    print(f"⏱️  Total Execution Time: {total_duration:.2f} seconds")
    print("==================================================")

if __name__ == "__main__":
    run_full_pipeline()