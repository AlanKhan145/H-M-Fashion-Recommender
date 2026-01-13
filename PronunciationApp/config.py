import json

def load_config(config_path="config.json"):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Không thể tải config: {e}")

config = load_config()

# Các biến cấu hình mặc định hoặc từ file
N_MFCC = config.get("N_MFCC", 13)
MAX_DISTANCE_FOR_100 = config.get("MAX_DISTANCE_FOR_100", 50.0)
MAX_DISTANCE_FOR_0 = config.get("MAX_DISTANCE_FOR_0", 300.0)
RECORD_SECONDS = config.get("RECORD_SECONDS", 3)
CHUNK = config.get("CHUNK", 1024)
SAMPLE_RATE = config.get("SAMPLE_RATE", 44100)
AI_ADJUSTMENT_THRESHOLD = config.get("AI_ADJUSTMENT_THRESHOLD", 60)
AI_ADJUSTMENT_VALUE = config.get("AI_ADJUSTMENT_VALUE", 5.0)
