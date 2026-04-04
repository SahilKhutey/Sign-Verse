import os
import sys
import torch
import numpy as np
import json
import time

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api_server.model_loader import ModelLoader

def main():
    print("=== SignVerse AI: Generative Avatar Prototype ===")
    
    loader = ModelLoader()
    
    # 1. Input Text
    input_text = "Hello, how are you today?"
    print(f"\n1. Input Text: '{input_text}'")
    
    # 2. Tokenization (Simulated)
    # In production, this uses the SignTokenizer
    tokenizer = loader.get_sign_tokenizer()
    tokens = tokenizer.encode(input_text)
    print(f"2. Tokens: {tokens}")
    
    # 3. Motion Generation
    print("3. Generating motion sequence...")
    # Load co-speech model
    try:
        model = loader.get_co_speech_model()
        num_frames = 30 # ~1 second at 30fps
        
        # Dummy audio/emotion for prototype
        dummy_audio = torch.randn(1, num_frames, 128)
        dummy_emotion = torch.zeros(1, 32)
        
        with torch.no_grad():
            motion_tensor = model(dummy_audio, dummy_emotion)
            motion = motion_tensor.squeeze(0).cpu().numpy()
            
        print(f"   [SYNC] Generated {len(motion)} frames of 225-dim keypoints.")
        print(f"   [SYNC] Sample (Frame 0, First 5): {motion[0][:5]}")
        
    except Exception as e:
        print(f"   [ERROR] Generation failed: {e}")
        return

    # 4. Save for Unity Visualization
    os.makedirs("exports", exist_ok=True)
    export_path = "exports/prototype_motion.json"
    with open(export_path, "w") as f:
        json.dump({
            "text": input_text,
            "tokens": tokens,
            "motion": motion.tolist(),
            "timestamp": time.time()
        }, f)
        
    print(f"\n4. Exported motion sequence to: {export_path}")
    print("\n=== Prototype Ready for XR Integration ===")

if __name__ == "__main__":
    main()
