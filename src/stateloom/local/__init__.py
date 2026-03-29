"""Local model support via Ollama."""

from stateloom.local.adapter import OllamaAdapter
from stateloom.local.client import OllamaClient, OllamaModelNotFoundError, OllamaResponse
from stateloom.local.hardware import HardwareInfo, detect_hardware, recommend_models
from stateloom.local.manager import OllamaManager

__all__ = [
    "OllamaAdapter",
    "OllamaClient",
    "OllamaManager",
    "OllamaModelNotFoundError",
    "OllamaResponse",
    "HardwareInfo",
    "detect_hardware",
    "recommend_models",
]
