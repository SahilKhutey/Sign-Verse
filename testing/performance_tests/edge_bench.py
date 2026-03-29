import onnxruntime as ort
import time
import numpy as np
import os
import psutil

def benchmark_model(model_path, num_iters=100):
    """
    Benchmarks an ONNX model for edge deployment metrics:
    - Average Latency (ms)
    - Peak Memory Usage (MB)
    - Model Size (MB)
    """
    print(f"Benchmarking {model_path}...")
    
    # Model size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    # Initialize session
    providers = ['CPUExecutionProvider'] # Edge baseline
    session = ort.InferenceSession(model_path, providers=providers)
    
    input_shape = (1, 64)
    mask_shape = (64, 64)
    
    # Warmup
    for _ in range(10):
        tokens = np.random.randint(0, 512, input_shape).astype(np.int64)
        mask = np.triu(np.ones(mask_shape), 1).astype(bool)
        session.run(None, {"tokens": tokens, "mask": mask})
    
    # Latency test
    latencies = []
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)
    
    for _ in range(num_iters):
        tokens = np.random.randint(0, 512, input_shape).astype(np.int64)
        mask = np.triu(np.ones(mask_shape), 1).astype(bool)
        start = time.perf_counter()
        session.run(None, {"tokens": tokens, "mask": mask})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    peak_mem = process.memory_info().rss / (1024 * 1024)
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print("\n[Edge Benchmark Results]")
    print(f"Model Path: {model_path}")
    print(f"Model Size: {size_mb:.2f} MB  (Target < 20MB: {'PASS' if size_mb < 20 else 'FAIL'})")
    print(f"Avg Latency: {avg_latency:.2f} ms  (Target < 50ms: {'PASS' if avg_latency < 50 else 'FAIL'})")
    print(f"P95 Latency: {p95_latency:.2f} ms")
    print(f"Peak Memory: {peak_mem:.2f} MB")
    print(f"Incremental Memory: {peak_mem - start_mem:.2f} MB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/optimized_sign_foundation.onnx")
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        benchmark_model(args.model)
    else:
        print(f"Model not found: {args.model}")
