import asyncio
import websockets
import json
import time

async def test_sign_to_speech():
    uri = "ws://localhost:8002/ws/translate/sign-to-speech"
    headers = {"X-User-ID": "test_user_123"}
    
    async with websockets.connect(uri, extra_headers=headers) as websocket:
        print(f"Connected to {uri}")
        
        # Send mock video bytes
        mock_frame = b"\x00" * 1024 
        await websocket.send(mock_frame)
        print("Sent mock frame")
        
        # Receive translation
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Received: {data['text']} (Audio length: {len(data['audio'])})")

async def test_text_to_sign():
    import requests
    url = "http://localhost:8002/api/translate/text-to-sign"
    payload = {
        "text": "How are you?",
        "style": "friendly",
        "output_format": "motion"
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"Text-to-Sign Success: {data['text']} -> {len(data['motion'])} frames")
    else:
        print(f"Text-to-Sign Failed: {response.text}")

if __name__ == "__main__":
    # Note: Requires the server to be running: python api/real_time_api.py
    print("Starting API local tests...")
    # asyncio.run(test_sign_to_speech()) # Uncomment to test WS
    # test_text_to_sign() # Uncomment to test POST
