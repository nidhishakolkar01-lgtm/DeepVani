"""
DeepVani Backend - Configuration Settings Module
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "DeepVani - Silent Speech Decoder & Neural Voice Platform"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Security Settings
    API_KEY: str = os.getenv("DEEPVANI_API_KEY", "deepvani-local-dev-key-2026")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # CORS Allowed Origins
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT: str = os.path.dirname(BASE_DIR)
    MODEL_PATH: str = os.path.join(BASE_DIR, "models", "silent_speech_model.pkl")
    STATIC_DIR: str = os.path.join(BASE_DIR, "static")
    AUDIO_DIR: str = os.path.join(STATIC_DIR, "audio")

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"


settings = Settings()
os.makedirs(settings.AUDIO_DIR, exist_ok=True)
