"""
Audio streaming utilities for handling real-time audio processing.
"""

import asyncio
import numpy as np
from typing import AsyncGenerator, Optional, Union
import logging

logger = logging.getLogger(__name__)


class AudioStreamer:
    """Handles streaming audio data for real-time processing."""
    
    def __init__(self, chunk_size: int = 16000, sample_rate: int = 16000):
        """
        Initialize the audio streamer.
        
        Args:
            chunk_size: Size of audio chunks in samples
            sample_rate: Audio sample rate in Hz
        """
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.is_streaming = False
        
    async def stream_audio(
        self, 
        audio_source: Union[str, bytes, np.ndarray]
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Stream audio data in chunks.
        
        Args:
            audio_source: Source of audio data
            
        Yields:
            Audio chunks as numpy arrays
        """
        try:
            if isinstance(audio_source, str):
                # File path
                async for chunk in self._stream_from_file(audio_source):
                    yield chunk
            elif isinstance(audio_source, bytes):
                # Raw bytes
                async for chunk in self._stream_from_bytes(audio_source):
                    yield chunk
            elif isinstance(audio_source, np.ndarray):
                # Numpy array
                async for chunk in self._stream_from_array(audio_source):
                    yield chunk
            else:
                raise ValueError(f"Unsupported audio source type: {type(audio_source)}")
                
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")
            raise
            
    async def _stream_from_file(self, file_path: str) -> AsyncGenerator[np.ndarray, None]:
        """Stream audio from file path."""
        # Placeholder implementation
        logger.info(f"Streaming audio from file: {file_path}")
        # This would be implemented with actual file reading logic
        yield np.zeros(self.chunk_size, dtype=np.float32)
        
    async def _stream_from_bytes(self, audio_bytes: bytes) -> AsyncGenerator[np.ndarray, None]:
        """Stream audio from raw bytes."""
        # Placeholder implementation
        logger.info(f"Streaming audio from bytes: {len(audio_bytes)} bytes")
        # This would be implemented with actual byte processing logic
        yield np.zeros(self.chunk_size, dtype=np.float32)
        
    async def _stream_from_array(self, audio_array: np.ndarray) -> AsyncGenerator[np.ndarray, None]:
        """Stream audio from numpy array."""
        logger.info(f"Streaming audio from array: {audio_array.shape}")
        
        # Split array into chunks
        for i in range(0, len(audio_array), self.chunk_size):
            chunk = audio_array[i:i + self.chunk_size]
            
            # Pad last chunk if necessary
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
                
            yield chunk
            await asyncio.sleep(0.001)  # Small delay to prevent blocking
            
    async def start_streaming(self):
        """Start the audio stream."""
        self.is_streaming = True
        logger.info("Audio streaming started")
        
    async def stop_streaming(self):
        """Stop the audio stream."""
        self.is_streaming = False
        logger.info("Audio streaming stopped")
        
    def get_streaming_status(self) -> bool:
        """Get current streaming status."""
        return self.is_streaming
