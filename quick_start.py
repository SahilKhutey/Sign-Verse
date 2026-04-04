import asyncio
import os
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from core.translation_engine import SignVerseTranslationEngine

async def main():
    print("--- SignVerse AI Quick Start ---")
    
    # Initialize engine
    engine = SignVerseTranslationEngine()
    
    # Quick test: Sign to Speech
    print("\n[Test 1] Sign to Speech Translation...")
    # In a real scenario, this would be a video path or bytes
    result = await engine.process_frame({"frames_processed": 0}, b"video_data")
    print(f"  Translation: {result['text']}")
    print(f"  Audio Samples: {len(result['speech'])} (first 5: {result['speech'][:5]})")
    
    # Quick test: Speech to Sign
    print("\n[Test 2] Speech to Sign Translation...")
    result = await engine.speech_to_sign(b"audio_bytes")
    print(f"  Recognized Text: {result['text']}")
    print(f"  Gesture Tokens: {result['tokens']}")
    
    # Real-time conversation simulation
    print("\n[Test 3] Simulated Real-time Session...")
    session = await engine.start_conversation_session("dev_user_01")
    for i in range(3):
        res = await engine.process_frame(session, b"frame_data")
        print(f"  Frame {i+1}: {res['text']}")

if __name__ == "__main__":
    asyncio.run(main())
