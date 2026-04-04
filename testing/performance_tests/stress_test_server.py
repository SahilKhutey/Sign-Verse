"""
Stress Test — Real-time Infernece Server
Simulates multiple concurrent WebSocket clients to measure latency and stability.
"""

import asyncio
import json
import time
import numpy as np
import websockets
import argparse
from typing import List, Dict

# Standard 225-dim keypoint vector for testing
DUMMY_KEYPOINTS = np.random.rand(225).tolist()

async def client_session(client_id: int, url: str, num_frames: int, fps: int):
    """Simulates a single AR/VR client streaming keypoints."""
    latencies = []
    
    try:
        async with websockets.connect(url) as ws:
            print(f"  [Client {client_id:02d}] Connected")
            
            for i in range(num_frames):
                start_time = time.perf_counter()
                
                payload = {
                    "keypoints": DUMMY_KEYPOINTS,
                    "text": "HELLO",
                    "frame_id": i
                }
                
                await ws.send(json.dumps(payload))
                
                # Receive response
                response = await ws.recv()
                data = json.loads(response)
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                # Maintain 30 FPS
                sleep_time = max(0, (1.0 / fps) - (end_time - start_time))
                await asyncio.sleep(sleep_time)
                
    except Exception as e:
        print(f"  [Client {client_id:02d}] Error: {e}")
        
    return latencies

async def run_stress_test(url: str, num_clients: int, num_frames: int, fps: int):
    print(f"\n{'='*60}")
    print(f"  SignVerse AI Stress Test")
    print(f"  URL:         {url}")
    print(f"  Clients:     {num_clients}")
    print(f"  Frames/cl:   {num_frames}")
    print(f"  Target FPS:  {fps}")
    print(f"{'='*60}\n")
    
    tasks = [client_session(i, url, num_frames, fps) for i in range(num_clients)]
    all_results = await asyncio.gather(*tasks)
    
    # Flatten all latencies
    flat_latencies = [l for client_res in all_results for l in client_res]
    
    if not flat_latencies:
        print("  ✗ No data collected")
        return

    mean_lat = np.mean(flat_latencies)
    p95_lat = np.percentile(flat_latencies, 95)
    p99_lat = np.percentile(flat_latencies, 99)
    min_lat = np.min(flat_latencies)
    max_lat = np.max(flat_latencies)
    
    print(f"\nResults (ms):")
    print(f"  Mean:   {mean_lat:8.2f}")
    print(f"  P95:    {p95_lat:8.2f}")
    print(f"  P99:    {p99_lat:8.2f}")
    print(f"  Min:    {min_lat:8.2f}")
    print(f"  Max:    {max_lat:8.2f}")
    
    success = p95_lat < 30.0
    print(f"\nTarget Performance (<30ms P95): {'PASS' if success else 'FAIL'}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/ws/stream")
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=30)
    
    args = parser.parse_args()
    
    asyncio.run(run_stress_test(args.url, args.clients, args.frames, args.fps))
