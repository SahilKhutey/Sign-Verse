import onnxruntime as ort
import time
import numpy as np
import os
import psutil
import json

class EdgeBenchmarker:
    """
    EdgeBenchmarker — Standardized performance metrics for optimized models.
    Measures latency, memory peak, and size across backends.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.size_mb = os.path.getsize(model_path) / (1024 * 1024)
        self.results = {}

    def run_benchmark(self, num_iters=100, backend='cpu'):
        print(f"Benchmarking {self.model_path} on {backend}...")
        
        providers = ['CPUExecutionProvider']
        if backend == 'npu': providers = ['XNNPACKExecutionProvider'] + providers
        
        session = ort.InferenceSession(self.model_path, providers=providers)
        
        tokens = np.random.randint(0, 512, (1, 64)).astype(np.int64)
        mask = np.triu(np.ones((64, 64)), 1).astype(bool)
        
        # Warmup (10%)
        for _ in range(10): session.run(None, {"tokens": tokens, "mask": mask})

        # Memory & Latency tracking
        latencies = []
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)
        
        for _ in range(num_iters):
            start = time.perf_counter()
            session.run(None, {"tokens": tokens, "mask": mask})
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        finish_mem = process.memory_info().rss / (1024 * 1024)
        
        avg_lat = np.mean(latencies)
        p95_lat = np.percentile(latencies, 95)
        
        self.results = {
            "model": os.path.basename(self.model_path),
            "size_mb": round(self.size_mb, 2),
            "avg_latency_ms": round(avg_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "peak_memory_mb": round(finish_mem, 2),
            "memory_usage_mb": round(finish_mem - start_mem, 2),
            "status": "PASS" if (self.size_mb < 20 and avg_lat < 50) else "FAIL"
        }
        
        print(f"Results: {self.results['avg_latency_ms']}ms | {self.results['size_mb']}MB")
        return self.results

    def save_report(self, report_path):
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Report saved to {report_path}")
