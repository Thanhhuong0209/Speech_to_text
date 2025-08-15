"""
Configuration management for STT system.

This module provides centralized configuration management using environment
variables and sensible defaults for all system components.
"""

import os
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


@dataclass
class Config:
    """
    Configuration class for STT system.
    
    All configuration values can be set via environment variables
    with sensible defaults provided.
    """
    
    # Whisper model configuration
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    
    # Audio processing configuration
    sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    audio_chunk_size: int = int(os.getenv("AUDIO_CHUNK_SIZE", "16000"))
    max_audio_length: int = int(os.getenv("MAX_AUDIO_LENGTH", "300"))
    
    # Noise reduction configuration
    noise_reduction_method: str = os.getenv("NOISE_REDUCTION_METHOD", "spectral")
    
    # LLM configuration
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")  # openai, local, huggingface
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    local_llm_url: str = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
    huggingface_model: str = os.getenv("HUGGINGFACE_MODEL", "gpt2")
    
    # Analysis configuration
    default_analysis_type: str = os.getenv("DEFAULT_ANALYSIS_TYPE", "summary")
    max_analysis_length: int = int(os.getenv("MAX_ANALYSIS_LENGTH", "1000"))
    analysis_timeout: int = int(os.getenv("ANALYSIS_TIMEOUT", "30"))
    
    # Web server configuration
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Cache configuration
    cache_dir: Path = Path(os.getenv("CACHE_DIR", "./cache"))
    max_cache_size: int = int(os.getenv("MAX_CACHE_SIZE", "1073741824"))  # 1GB
    
    # Logging configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Optional[str] = os.getenv("LOG_FILE")
    
    # LibriSpeech configuration
    librispeech_base_url: str = os.getenv(
        "LIBRISPEECH_BASE_URL", 
        "https://www.openslr.org/resources/12/"
    )
    max_concurrent_downloads: int = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "5"))
    
    # Performance configuration
    max_workers: int = int(os.getenv("MAX_WORKERS", "4"))
    chunk_timeout: int = int(os.getenv("CHUNK_TIMEOUT", "30"))
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._ensure_directories()
    
    def _validate_config(self):
        """Validate configuration values."""
        # Validate Whisper model size
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if self.whisper_model_size not in valid_models:
            raise ValueError(f"Invalid Whisper model size: {self.whisper_model_size}")
        
        # Validate sample rate
        if self.sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive: {self.sample_rate}")
        
        # Validate chunk size
        if self.audio_chunk_size <= 0:
            raise ValueError(f"Audio chunk size must be positive: {self.audio_chunk_size}")
        
        # Validate max audio length
        if self.max_audio_length <= 0:
            raise ValueError(f"Max audio length must be positive: {self.max_audio_length}")
        
        # Validate port
        if not (1 <= self.port <= 65535):
            raise ValueError(f"Port must be between 1 and 65535: {self.port}")
        
        # Validate cache size
        if self.max_cache_size <= 0:
            raise ValueError(f"Max cache size must be positive: {self.max_cache_size}")
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def get_whisper_config(self) -> dict:
        """Get Whisper-specific configuration."""
        return {
            "model_size": self.whisper_model_size,
            "sample_rate": self.sample_rate,
            "max_length": self.max_audio_length
        }
    
    def get_audio_config(self) -> dict:
        """Get audio processing configuration."""
        return {
            "sample_rate": self.sample_rate,
            "chunk_size": self.audio_chunk_size,
            "max_length": self.max_audio_length
        }
    
    def get_web_config(self) -> dict:
        """Get web server configuration."""
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug
        }
    
    def get_cache_config(self) -> dict:
        """Get cache configuration."""
        return {
            "cache_dir": str(self.cache_dir),
            "max_size": self.max_cache_size
        }
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "whisper_model_size": self.whisper_model_size,
            "sample_rate": self.sample_rate,
            "audio_chunk_size": self.audio_chunk_size,
            "max_audio_length": self.max_audio_length,
            "noise_reduction_method": self.noise_reduction_method,
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "cache_dir": str(self.cache_dir),
            "max_cache_size": self.max_cache_size,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "librispeech_base_url": self.librispeech_base_url,
            "max_concurrent_downloads": self.max_concurrent_downloads,
            "max_workers": self.max_workers,
            "chunk_timeout": self.chunk_timeout
        }
    
    def update_from_env(self):
        """Update configuration from environment variables."""
        # This method can be used to dynamically update config
        # from environment variables at runtime
        pass
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls()
    
    @classmethod
    def from_file(cls, config_file: str) -> 'Config':
        """Create configuration from file (future enhancement)."""
        # This would load from JSON/YAML config file
        raise NotImplementedError("File-based configuration not yet implemented")


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance."""
    return config


def reload_config() -> Config:
    """Reload configuration from environment."""
    global config
    config = Config()
    return config


# Configuration validation tests
if __name__ == "__main__":
    # Test configuration creation
    test_config = Config()
    print("Configuration created successfully:")
    print(f"Whisper model: {test_config.whisper_model_size}")
    print(f"Sample rate: {test_config.sample_rate}")
    print(f"Port: {test_config.port}")
    
    # Test configuration validation
    try:
        # This should raise an error
        invalid_config = Config(whisper_model_size="invalid")
    except ValueError as e:
        print(f"Validation error caught: {e}")
    
    # Test configuration serialization
    config_dict = test_config.to_dict()
    print(f"Configuration as dict: {config_dict}")
    
    print("Configuration tests passed!")
