import asyncio
import websockets
import json
import time
import numpy as np
import cv2
import argparse

async def simulate_client(client_id, url, num_frames=50):
    async with websockets.connect(url) as websocket:
        print(f"Client {client_id} connected.")
        
        latencies = []
        for i in range(num_frames):
            try:
                # Create a dummy frame (640x480 RGB)
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', frame)
                
                start_time = time.perf_counter()
                await websocket.send(buffer.tobytes())
                
                response = await websocket.recv()
                latency = (time.perf_counter() - start_time) * 1000
                latencies.append(latency)
                
                if i % 5 == 0:
                    data = json.loads(response)
                    print(f"Client {client_id} - Frame {i}: {data.get('text', 'N/A')} ({latency:.1f}ms)")
            except Exception as e:
                print(f"Client {client_id} error on frame {i}: {e}")
                break
        
        avg_latency = sum(latencies) / len(latencies)
        print(f"Client {client_id} finished. Avg Latency: {avg_latency:.2f}ms")
        return avg_latency

async def run_stress_test(url, num_clients=5, frames_per_client=50):
    print(f"Starting stress test with {num_clients} clients...")
    tasks = [simulate_client(i, url, frames_per_client) for i in range(num_clients)]
    results = await asyncio.gather(*tasks)
    
    total_avg = sum(results) / len(results)
    print(f"\n--- Stress Test Results ---")
    print(f"Total Clients: {num_clients}")
    print(f"Average System Latency: {total_avg:.2f}ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="ws://localhost:8000/translate")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--frames", type=int, default=50)
    args = parser.parse_args()
    
    asyncio.run(run_stress_test(args.url, args.clients, args.frames))
