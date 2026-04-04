import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATHS = {
    "gesture_classifier": os.path.join(BASE_DIR, "models/gesture_model/gesture_classifier.pt"),
    "speech_model": "base"
}

API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000
}
