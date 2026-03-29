import os
import time
import subprocess
import shutil
from datetime import datetime

PENDING_DIR = "datasets/user_feedback/pending_validation"
VALIDATED_DIR = "datasets/user_feedback/validated"
THRESHOLD = 5  # Trigger training after 5 new corrections

os.makedirs(PENDING_DIR, exist_ok=True)
os.makedirs(VALIDATED_DIR, exist_ok=True)

def monitor_and_train():
    print(f"[{datetime.now()}] SignVerse Automated Trainer Watchdog Started.")
    print(f"Monitoring: {PENDING_DIR}")
    
    while True:
        pending_files = [f for f in os.listdir(PENDING_DIR) if f.endswith(".json")]
        
        if len(pending_files) >= THRESHOLD:
            print(f"[{datetime.now()}] Threshold reached ({len(pending_files)} samples). Starting validation & training...")
            
            # Step 1: "Validate" samples (move to validated dir)
            for f in pending_files:
                src = os.path.join(PENDING_DIR, f)
                dst = os.path.join(VALIDATED_DIR, f)
                shutil.move(src, dst)
            
            # Step 2: Trigger Fine-Tuning
            print(f"[{datetime.now()}] Executing EWC Fine-Tuning cycle...")
            try:
                # Run the training script as a subprocess
                result = subprocess.run([
                    "python", "training/train_continuous_model.py", 
                    "--mode", "fine-tune",
                    "--epochs", "3",
                    "--importance", "1500"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"[{datetime.now()}] Fine-tuning successful!")
                    print(result.stdout)
                else:
                    print(f"[{datetime.now()}] Fine-tuning failed with exit code {result.returncode}")
                    print(result.stderr)
                    
            except Exception as e:
                print(f"[{datetime.now()}] Error during training trigger: {e}")
        
        # Sleep for a bit before checking again
        time.sleep(30) # Check every 30 seconds

if __name__ == "__main__":
    monitor_and_train()
