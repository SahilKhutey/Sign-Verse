import requests
import json
import time

def test_backend_translation():
    print("--- SignVerse Backend-AI Integration Test ---")
    url = "http://localhost:8001/api/translations/translate"
    
    # 1. Test Sign -> Speech
    print("\n[Test 1] POST /api/translations/translate (Input: Sign)")
    payload = {
        "user_id": 1,
        "data": "base64_mock_video_data",
        "input_type": "sign",
        "target_lang": "en"
    }
    
    try:
        resp = requests.post(url, json=payload)
        if resp.status_code == 200:
            result = resp.json()
            print(f"  Success! Translated Text: {result.get('text')}")
            print(f"  Confidence: {result.get('confidence')}")
        else:
            print(f"  Failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"  Error connecting to backend: {e}")

    # 2. Test Audio -> Sign
    print("\n[Test 2] POST /api/translations/translate (Input: Audio)")
    payload["input_type"] = "audio"
    payload["data"] = "base64_mock_audio_data"
    
    try:
        resp = requests.post(url, json=payload)
        if resp.status_code == 200:
            result = resp.json()
            print(f"  Success! Recognized Text: {result.get('text')}")
            print(f"  Tokens: {result.get('tokens')}")
        else:
            print(f"  Failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    # Wait a bit for backend to start if needed
    time.sleep(2)
    test_backend_translation()
