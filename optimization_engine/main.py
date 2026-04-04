import argparse
import os
import json
import torch
from optimization_engine.adaptive_architectures import get_edge_architecture
from optimization_engine.compressor import EdgeCompressor
from optimization_engine.benchmarker import EdgeBenchmarker
from models.model_encryptor import encrypt_model

def run_optimization_engine(model_name, pruning_amount=0.8):
    """
    Unified Optimization Lifecycle:
    Prune → Quantize → Encrypt → Benchmark
    """
    print(f"\n{'='*60}")
    print(f"  SignVerse — Unified Edge AI Optimization Engine")
    print(f"{'='*60}\n")
    
    # 1. Config & Paths
    base_dir = "models"
    input_path = os.path.join(base_dir, model_name)
    output_onnx = os.path.join(base_dir, f"optimized_{model_name.replace('.pt', '.onnx')}")
    secure_bin = os.path.join(base_dir, f"optimized_{model_name.replace('.pt', '_secure.bin')}")
    report_path = os.path.join(base_dir, f"optimization_report_{model_name.replace('.pt', '.json')}")

    if not os.path.exists(input_path):
        print(f"Error: Model not found at {input_path}")
        return

    # 2. Architecture Selection
    # For now, we assume all large SignVerse models are Transformers
    model_type = "transformer"
    print(f"Detecting architecture: {model_type.upper()}")
    
    # Initialize Edge-Lite Model
    edge_model = get_edge_architecture(model_type, num_layers=4)
    
    # 3. Compression Lifecycle (Prune + Quantize)
    compressor = EdgeCompressor(edge_model, input_path, output_onnx)
    compressor.prune_and_quantize(pruning_amount=pruning_amount)

    # 4. Security Layer (AES-256 Encryption)
    encrypt_model(output_onnx, secure_bin)

    # 5. Benchmarking & Verification
    benchmarker = EdgeBenchmarker(output_onnx)
    report = benchmarker.run_benchmark(num_iters=100)
    benchmarker.save_report(report_path)

    print(f"\n{'='*60}")
    print(f"  ✓ Optimization Lifecycle Complete")
    print(f"  Final Size:   {report['size_mb']} MB")
    print(f"  Avg Latency:  {report['avg_latency_ms']} ms")
    print(f"  Status:       {report['status']}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignVerse Optimization Engine")
    parser.add_argument("--model", default="foundation_model_best.pt", help="Model filename in /models")
    parser.add_argument("--prune", type=float, default=0.8, help="Pruning ratio (0-1)")
    
    args = parser.parse_args()
    run_optimization_engine(args.model, args.prune)
