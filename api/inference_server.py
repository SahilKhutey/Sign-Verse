import os
import sys
import time
import torch
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
import uvicorn

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api_server.model_loader import ModelLoader

app = FastAPI(title="SignVerse AI API")
loader = ModelLoader()

# Move extractor to global to avoid re-initialization latency
from vision_pipeline.feature_extractor import FeatureExtractor
extractor = FeatureExtractor()

# Global metrics
METRICS = {
    "latency_rec": 0.0,
    "latency_trans": 0.0,
    "latency_gen": 0.0,
    "active_connections": 0,
    "start_time": time.time()
}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "models_loaded": {
            "sign_transformer": loader.get_sign_transformer() is not None,
            "gesture_model": loader.get_gesture_model() is not None,
            "co_speech_model": loader.get_co_speech_model() is not None
        }
    }

@app.get("/metrics")
async def get_metrics():
    import psutil
    gpu_metrics = {"load": 0, "mem": 0}
    try:
        # Simplistic GPU metric fetch (requires pynvml in real prod)
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_metrics["load"] = util.gpu
        gpu_metrics["mem"] = info.used // (1024**2)
    except:
        pass
        
    return {
        "uptime": time.time() - METRICS["start_time"],
        "status": "healthy",
        "latency_rec": METRICS["latency_rec"],
        "latency_trans": METRICS["latency_trans"],
        "latency_gen": METRICS["latency_gen"],
        "active_connections": METRICS["active_connections"],
        "gpu_load": gpu_metrics["load"],
        "gpu_mem": gpu_metrics["mem"]
    }

@app.post("/generate/motion")
async def generate_motion(request: Request):
    """Generative co-speech motion synthesis (Text -> Motion)."""
    data = await request.json()
    tokens = data.get("tokens", [])
    num_frames = data.get("frames", 30)
    
    # Load co-speech model
    model = loader.get_co_speech_model()
    
    try:
        start_gen = time.perf_counter()
        # Dummy audio feature extraction (normally from binary audio input)
        # Using 128-dim features as per ModelLoader default
        dummy_audio = torch.randn(1, num_frames, 128)
        dummy_emotion = torch.zeros(1, 64) # Neutral emotion
        
        with torch.no_grad():
            # forward(audio_features, emotion_embedding)
            motion_tensor = model(dummy_audio, dummy_emotion)
            # motion_tensor shape: [1, num_frames, 225]
            motion = motion_tensor.squeeze(0).cpu().numpy().tolist()
            
        METRICS["latency_gen"] = (time.perf_counter() - start_gen) * 1000
        return {"motion": motion, "frames": num_frames}
    except Exception as e:
        return {"error": str(e)}

def load_production_model(model_type: str):
    """Bridge to the existing ModelLoader."""
    if model_type == "sign_to_text":
        return loader.get_sign_transformer()
    elif model_type == "gesture":
        return loader.get_gesture_model()
    # Add other mappings as needed
    return None

def decode_video(video_bytes: bytes):
    """Decode raw video bytes into a list of frames (numpy arrays)."""
    # For a real-time stream of individual frames sent as bytes:
    arr = np.frombuffer(video_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is not None:
        return [frame]
    return []

@app.websocket("/translate")
async def translate_sign(websocket: WebSocket):
    await websocket.accept()
    METRICS["active_connections"] += 1
    
    # Pre-load the model to avoid latency on first frame
    model = load_production_model("sign_to_text")
    
    # Accept language preference from client
    query_params = websocket.query_params
    target_lang = query_params.get("target_lang", "ASL")
    
    try:
        while True:
            # Receive video frames via WebSocket
            data = await websocket.receive()
            
            if "bytes" in data:
                video_data = data["bytes"]
            elif "text" in data:
                # Handle potential JSON messages for metadata switching
                continue
            else:
                continue
            
            start_time = time.perf_counter()
            
            # Process in real-time
            frames = decode_video(video_data)
            
            if not frames:
                continue
            
            # Translation logic
            # Use global extractor
            all_features = []
            for frame in frames:
                try:
                    start_rec = time.perf_counter()
                    features, _, _ = extractor.extract(frame)
                    all_features.append(features)
                    METRICS["latency_rec"] = (time.perf_counter() - start_rec) * 1000
                except Exception as e:
                    print(f"Extraction error: {e}")
                    continue
            
            seq = torch.tensor(np.array(all_features), dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                start_trans = time.perf_counter()
                # Multilingual translation conditioning
                translation_ids = model.translate(seq, target_lang=target_lang)
                tokenizer = loader.get_sign_tokenizer()
                translation = tokenizer.decode(translation_ids[0]) if translation_ids else ""
                METRICS["latency_trans"] = (time.perf_counter() - start_trans) * 1000
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Send back translation
            await websocket.send_json({
                "text": translation,
                "confidence": 0.95, # Placeholder as model might not provide direct score
                "processing_time": processing_time
            })
            
    except WebSocketDisconnect:
        METRICS["active_connections"] -= 1
        print("Client disconnected")
    except Exception:
        METRICS["active_connections"] -= 1
        import traceback
        print(f"Error in translation loop:\n{traceback.format_exc()}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
