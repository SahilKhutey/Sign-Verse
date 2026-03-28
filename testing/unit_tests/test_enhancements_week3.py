import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from metrics.advanced_metrics import SignLanguageEvaluator
from preprocessor.video_pipeline import UnifiedPreprocessor

def test_metrics():
    print("\n--- Testing SignLanguageEvaluator ---")
    evaluator = SignLanguageEvaluator()
    
    # Mock data
    gen_text = "HELLO THANK YOU"
    target_text = "HELLO THANK YOU"
    
    # Mock gestures (N, T, D)
    gen_gestures = torch.randn(1, 30, 225)
    real_gestures = torch.randn(1, 30, 225)
    
    model_output = {
        "text": gen_text,
        "gestures": gen_gestures
    }
    ground_truth = {
        "text": target_text,
        "gestures": real_gestures
    }
    
    results = evaluator.evaluate_pipeline(model_output, ground_truth)
    print("Metrics Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    
    assert results["translation_wer"] == 0.0
    print("✓ Metrics test passed")

def test_preprocessor():
  print("\n--- Testing UnifiedPreprocessor ---")
  try:
    # We won't run extraction as it needs real videos, but we can test the structure
    preprocessor = UnifiedPreprocessor()
    print("✓ Preprocessor initialization passed")
    
    # Test batch processing with dummy list (should fail gracefully or handle empty)
    tokens = preprocessor.process_batch([])
    assert tokens == []
    print("✓ Empty batch handling passed")
    
  except Exception as e:
    print(f"✗ Preprocessor test failed: {e}")

if __name__ == "__main__":
    test_metrics()
    test_preprocessor()
