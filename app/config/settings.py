import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # API settings
    port: int = int(os.getenv("PORT", 8000))
    
    # OCR settings
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "tesseract")
    
    # Logging settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env" 