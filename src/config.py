import os

from dotenv import load_dotenv

load_dotenv()

APP_TITLE = "Transcritor de Áudios com Whisper"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
DEFAULT_MODEL = "small"

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "true").lower() == "true"

_default_num_speakers = os.getenv("DEFAULT_NUM_SPEAKERS", "").strip()
DEFAULT_NUM_SPEAKERS = int(_default_num_speakers) if _default_num_speakers.isdigit() else None