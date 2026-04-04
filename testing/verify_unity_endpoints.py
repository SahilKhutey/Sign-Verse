import asyncio
import websockets
import httpx
import json

async def test_text_to_sign():
    print("--- Testing REST: /translate/text-to-sign ---")
    url = "http://localhost:8000/translate/text-to-sign"
    payload = {"text": "Hello world", "language": "ASL"}
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert "tokens" in response.json()

async def test_websocket_stream():
    print("\n--- Testing WebSocket: /ws/stream ---")
    uri = "ws://localhost:8000/ws/stream?target_lang=DGS"
    async with websockets.connect(uri) as websocket:
        # Send a mock keypoint frame (minimal representation)
        mock_data = {
            "type": "frame",
            "keypoints": [0.1] * 75  # 25 points * 3 dims
        }
        await websocket.send(json.dumps(mock_data))
        
        # Wait for response
        response = await websocket.recv()
        print(f"Received: {response}")
        res_json = json.loads(response)
        assert "status" in res_json or "gesture" in res_json

if __name__ == "__main__":
    asyncio.run(test_text_to_sign())
    try:
        asyncio.run(test_websocket_stream())
    except Exception as e:
        print(f"WebSocket Test Failed (Server might not be running or loopback issues): {e}")
