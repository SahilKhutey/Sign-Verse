import time
import torch
import numpy as np
import os
import sys

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api_server.model_loader import ModelLoader

class ModelBenchmark:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.device = next(model.parameters()).device

    def measure_latency(self, num_runs=50):
        latencies = []
        # Detect input size from model
        input_size = 225
        if hasattr(self.model, "lstm"):
            input_size = self.model.lstm.input_size
        elif hasattr(self.model, "feature_dim"):
            input_size = self.model.feature_dim
            
        if "gesture" in self.name:
            input_data = torch.randn(1, 30, input_size).to(self.device)
        elif "transformer" in self.name:
            input_data = torch.randn(1, 60, input_size).to(self.device)
        else:
            input_data = torch.randn(1, 1, input_size).to(self.device)

        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                if hasattr(self.model, "translate"):
                    _ = self.model.translate(input_data)
                else:
                    _ = self.model(input_data)
                latencies.append((time.perf_counter() - start) * 1000)
        
        return np.mean(latencies), np.percentile(latencies, 95)

    def measure_memory_usage(self):
        params = sum(p.numel() for p in self.model.parameters())
        # Simplistic estimate: params * 4 bytes (FP32)
        return params * 4 / (1024**2) 

def run_benchmarks():
    print("--------------------------------------------------------------")
    print("  SignVerse AI Performance Benchmarks")
    print("--------------------------------------------------------------")
    
    loader = ModelLoader()
    models_to_test = [
        ("gesture_recognition", loader.get_gesture_model()),
        ("sign_to_text_transformer", loader.get_sign_transformer()),
    ]
    
    for name, model in models_to_test:
        bench = ModelBenchmark(model, name)
        avg_lat, p95_lat = bench.measure_latency()
        mem = bench.measure_memory_usage()
        
        print(f"Model: {name}")
        print(f"  Avg Latency: {avg_lat:8.2f} ms")
        print(f"  P95 Latency: {p95_lat:8.2f} ms")
        print(f"  Memory (est): {mem:8.2f} MB")
        print("-" * 30)

if __name__ == "__main__":
    run_benchmarks()
