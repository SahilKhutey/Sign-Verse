import sys
import os

modules_to_test = [
    "ai_engine",
    "ai_models",
    "ai_models.gesture_recognition",
    "api_server",
    "backend",
    "common",
    "gesture_recognition",
    "nlp_translation",
    "training",
    "vision_pipeline"
]

results = {}

for mod in modules_to_test:
    try:
        __import__(mod)
        results[mod] = "OK"
    except ImportError as e:
        results[mod] = f"FAILED: {e}"
    except Exception as e:
        results[mod] = f"ERROR: {e}"

for mod, res in results.items():
    print(f"{mod}: {res}")

if all(res == "OK" for res in results.values()):
    sys.exit(0)
else:
    sys.exit(1)
