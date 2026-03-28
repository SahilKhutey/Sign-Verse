import time
import requests
import os
import sys

def get_metrics(url):
    try:
        resp = requests.get(f"{url}/metrics", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

def main():
    url = os.environ.get("SIGNVERSE_API_URL", "http://localhost:8888")
    print(f"--- SignVerse AI Real-time Monitor ---")
    print(f"Target: {url}")
    print("-" * 40)
    
    try:
        while True:
            metrics = get_metrics(url)
            if metrics:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"SignVerse AI Health: {metrics.get('status', 'OFFLINE')}")
                print(f"Uptime: {metrics.get('uptime', 'N/A')}")
                print("-" * 20)
                print(f"Latencies (ms):")
                print(f"  Recognition: {metrics.get('latency_rec', 0):.2f}")
                print(f"  Translation: {metrics.get('latency_trans', 0):.2f}")
                print(f"  Generation:  {metrics.get('latency_gen', 0):.2f}")
                print("-" * 20)
                print(f"System:")
                print(f"  GPU Load:    {metrics.get('gpu_load', 0)}%")
                print(f"  GPU Mem:     {metrics.get('gpu_mem', 0)} MB")
                print(f"  Connections: {metrics.get('active_connections', 0)}")
            else:
                print(".", end="", flush=True)
            
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")

if __name__ == "__main__":
    main()
