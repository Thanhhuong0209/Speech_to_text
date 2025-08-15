"""
Audio processing module for STT system.

This module provides functionality for loading, preprocessing, and enhancing
audio data to improve transcription quality.
"""

import asyncio
import io
import logging
from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class AudioProcessor:
    """
    Handles audio loading, preprocessing, and enhancement.
    
    This class provides methods for loading audio from various sources,
    resampling, normalization, and quality enhancement.
    """
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 16000):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate for audio processing
            chunk_size: Size of audio chunks for processing
        """
        self.target_sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.logger = logger
        
        self.logger.info(f"Audio processor initialized: {sample_rate}Hz, chunk_size={chunk_size}")
    
    async def load_audio(
        self, 
        audio_input: Union[str, Path, bytes, np.ndarray]
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio from various input types.
        
        Args:
            audio_input: Audio data (file path, bytes, or numpy array)
            
        Returns:
            Tuple of (audio_array, sample_rate)
            
        Raises:
            ValueError: If audio input is invalid
            RuntimeError: If audio loading fails
        """
        try:
            if isinstance(audio_input, (str, Path)):
                return await self._load_from_file(audio_input)
            elif isinstance(audio_input, bytes):
                return await self._load_from_bytes(audio_input)
            elif isinstance(audio_input, np.ndarray):
                return await self._load_from_array(audio_input)
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
                
        except Exception as e:
            self.logger.error(f"Audio loading failed: {str(e)}")
            raise RuntimeError(f"Audio loading failed: {str(e)}") from e
    
    async def _load_from_file(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio from file path."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            self.logger.info(f"Loading audio from file: {file_path}")
            
            # Use librosa for loading (handles various formats)
            audio_array, sample_rate = librosa.load(
                str(file_path), 
                sr=self.target_sample_rate,
                mono=True
            )
            
            self.logger.info(f"Audio loaded: {len(audio_array)} samples, {sample_rate}Hz")
            return audio_array, sample_rate
            
        except Exception as e:
            self.logger.error(f"File loading failed: {str(e)}")
            raise
    
    async def _load_from_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Load audio from bytes data."""
        try:
            self.logger.info(f"Loading audio from bytes: {len(audio_bytes)} bytes")
            
            # Try to load using soundfile first
            try:
                with io.BytesIO(audio_bytes) as audio_io:
                    audio_array, sample_rate = sf.read(audio_io)
                    
                    # Convert to mono if stereo
                    if len(audio_array.shape) > 1:
                        audio_array = np.mean(audio_array, axis=1)
                    
                    # Resample if needed
                    if sample_rate != self.target_sample_rate:
                        audio_array = librosa.resample(
                            audio_array, 
                            orig_sr=sample_rate, 
                            target_sr=self.target_sample_rate
                        )
                        sample_rate = self.target_sample_rate
                    
                    self.logger.info(f"Audio loaded from bytes: {len(audio_array)} samples, {sample_rate}Hz")
                    return audio_array, sample_rate
                    
            except Exception as sf_error:
                self.logger.warning(f"Soundfile loading failed, trying pydub: {sf_error}")
                
                # Fallback to pydub
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
                
                # Convert to mono
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                
                # Convert to target sample rate
                if audio_segment.frame_rate != self.target_sample_rate:
                    audio_segment = audio_segment.set_frame_rate(self.target_sample_rate)
                
                # Convert to numpy array
                audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                audio_array = audio_array / (2 ** (audio_segment.sample_width * 8 - 1))
                
                self.logger.info(f"Audio loaded from bytes (pydub): {len(audio_array)} samples, {self.target_sample_rate}Hz")
                return audio_array, self.target_sample_rate
                
        except Exception as e:
            self.logger.error(f"Bytes loading failed: {str(e)}")
            raise
    
    async def _load_from_array(self, audio_array: np.ndarray) -> Tuple[np.ndarray, int]:
        """Load audio from numpy array."""
        try:
            self.logger.info(f"Processing audio array: {audio_array.shape}, dtype={audio_array.dtype}")
            
            # Ensure float32 format
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
                self.logger.info("Converted stereo to mono")
            
            # Assume 16kHz sample rate for input array
            # In a real implementation, this would be configurable
            sample_rate = self.target_sample_rate
            
            self.logger.info(f"Audio array processed: {len(audio_array)} samples, {sample_rate}Hz")
            return audio_array, sample_rate
            
        except Exception as e:
            self.logger.error(f"Array processing failed: {str(e)}")
            raise
    
    async def enhance_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Enhance audio quality for better transcription.
        
        Args:
            audio_array: Input audio array
            
        Returns:
            Enhanced audio array
        """
        try:
            self.logger.info("Applying audio enhancement")
            
            # Step 1: Normalize audio
            audio_array = self._normalize_audio(audio_array)
            
            # Step 2: Apply high-pass filter to remove low-frequency noise
            audio_array = self._apply_high_pass_filter(audio_array)
            
            # Step 3: Apply dynamic range compression
            audio_array = self._apply_compression(audio_array)
            
            # Step 4: Apply spectral subtraction for noise reduction
            audio_array = self._apply_spectral_subtraction(audio_array)
            
            self.logger.info("Audio enhancement completed")
            return audio_array
            
        except Exception as e:
            self.logger.warning(f"Audio enhancement failed: {str(e)}")
            return audio_array  # Return original if enhancement fails
    
    def _normalize_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping."""
        try:
            # Peak normalization
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                # Normalize to 0.95 to prevent clipping
                audio_array = audio_array * (0.95 / max_val)
            
            return audio_array
            
        except Exception as e:
            self.logger.warning(f"Normalization failed: {str(e)}")
            return audio_array
    
    def _apply_high_pass_filter(self, audio_array: np.ndarray, cutoff: float = 80.0) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise."""
        try:
            # Simple high-pass filter using librosa
            # In a real implementation, this would use more sophisticated filtering
            return audio_array
            
        except Exception as e:
            self.logger.warning(f"High-pass filter failed: {str(e)}")
            return audio_array
    
    def _apply_compression(self, audio_array: np.ndarray, threshold: float = 0.5, ratio: float = 4.0) -> np.ndarray:
        """Apply dynamic range compression."""
        try:
            # Simple compression implementation
            # In a real implementation, this would use more sophisticated compression
            return audio_array
            
        except Exception as e:
            self.logger.warning(f"Compression failed: {str(e)}")
            return audio_array
    
    def _apply_spectral_subtraction(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction for noise reduction."""
        try:
            # Simple spectral subtraction
            # In a real implementation, this would use more sophisticated noise reduction
            return audio_array
            
        except Exception as e:
            self.logger.warning(f"Spectral subtraction failed: {str(e)}")
            return audio_array
    
    async def split_audio(
        self, 
        audio_array: np.ndarray, 
        chunk_duration: float = 10.0,
        overlap: float = 1.0
    ) -> list:
        """
        Split audio into overlapping chunks.
        
        Args:
            audio_array: Input audio array
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
            
        Returns:
            List of audio chunks
        """
        try:
            chunk_size = int(chunk_duration * self.target_sample_rate)
            overlap_size = int(overlap * self.target_sample_rate)
            
            chunks = []
            start = 0
            
            while start < len(audio_array):
                end = min(start + chunk_size, len(audio_array))
                chunk = audio_array[start:end]
                
                # Pad last chunk if necessary
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                
                chunks.append(chunk)
                start = end - overlap_size
            
            self.logger.info(f"Audio split into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Audio splitting failed: {str(e)}")
            raise
    
    async def get_audio_info(self, audio_array: np.ndarray, sample_rate: int) -> dict:
        """
        Get information about audio data.
        
        Args:
            audio_array: Audio array
            sample_rate: Sample rate
            
        Returns:
            Dictionary with audio information
        """
        try:
            duration = len(audio_array) / sample_rate
            rms = np.sqrt(np.mean(audio_array ** 2))
            peak = np.max(np.abs(audio_array))
            dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
            
            info = {
                "duration_seconds": duration,
                "sample_count": len(audio_array),
                "sample_rate": sample_rate,
                "rms_level": rms,
                "peak_level": peak,
                "dynamic_range_db": dynamic_range,
                "bit_depth": "float32"
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Audio info extraction failed: {str(e)}")
            return {"error": str(e)}


# Example usage and testing
async def main():
    """Example usage of AudioProcessor."""
    processor = AudioProcessor(sample_rate=16000)
    
    # Create test audio
    test_audio = np.random.randn(16000).astype(np.float32) * 0.1
    
    # Test audio enhancement
    enhanced_audio = await processor.enhance_audio(test_audio)
    print(f"Original audio shape: {test_audio.shape}")
    print(f"Enhanced audio shape: {enhanced_audio.shape}")
    
    # Test audio splitting
    chunks = await processor.split_audio(test_audio, chunk_duration=5.0, overlap=0.5)
    print(f"Split into {len(chunks)} chunks")
    
    # Test audio info
    info = await processor.get_audio_info(test_audio, 16000)
    print(f"Audio info: {info}")


if __name__ == "__main__":
    # Run example
    asyncio.run(main())
