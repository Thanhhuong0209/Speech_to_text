"""
Speech-to-Text (STT) System with Direct Web Data Ingestion

This package provides a complete STT solution that can process audio data
directly from web sources without requiring local storage of large datasets.
"""

__version__ = "1.0.0"
__author__ = "AI Intern Assessment"
__description__ = "STT System with Web Data Ingestion"

from .core.stt_engine import STTEngine
from .data.librispeech_ingester import LibriSpeechIngester
from .models.whisper_wrapper import WhisperWrapper

__all__ = [
    "STTEngine",
    "LibriSpeechIngester", 
    "WhisperWrapper"
]
