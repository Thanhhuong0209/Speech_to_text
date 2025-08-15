"""
Noise reduction module for STT system.

This module provides various noise reduction techniques to improve
audio quality and transcription accuracy.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import numpy as np
import librosa

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class NoiseReducer:
    """
    Implements various noise reduction techniques for audio enhancement.
    
    This class provides methods for reducing different types of noise
    including stationary noise, impulse noise, and environmental noise.
    """
    
    def __init__(self, reduction_method: str = "spectral"):
        """
        Initialize noise reducer.
        
        Args:
            reduction_method: Method to use for noise reduction
        """
        self.method = reduction_method
        self.logger = logger
        
        # Supported methods
        self.supported_methods = [
            "spectral",      # Spectral subtraction
            "wiener",        # Wiener filtering
            "vad",          # Voice Activity Detection
            "adaptive",      # Adaptive noise reduction
            "combined"       # Combined approach
        ]
        
        if reduction_method not in self.supported_methods:
            self.logger.warning(f"Unsupported method: {reduction_method}, using spectral")
            self.method = "spectral"
        
        self.logger.info(f"Noise reducer initialized with method: {self.method}")
    
    async def reduce_noise(
        self, 
        audio_array: np.ndarray, 
        sample_rate: int,
        **kwargs
    ) -> np.ndarray:
        """
        Reduce noise in audio using the specified method.
        
        Args:
            audio_array: Input audio array
            sample_rate: Sample rate of the audio
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Noise-reduced audio array
        """
        try:
            self.logger.info(f"Applying noise reduction: {self.method}")
            
            if self.method == "spectral":
                return await self._spectral_subtraction(audio_array, sample_rate, **kwargs)
            elif self.method == "wiener":
                return await self._wiener_filtering(audio_array, sample_rate, **kwargs)
            elif self.method == "vad":
                return await self._voice_activity_detection(audio_array, sample_rate, **kwargs)
            elif self.method == "adaptive":
                return await self._adaptive_noise_reduction(audio_array, sample_rate, **kwargs)
            elif self.method == "combined":
                return await self._combined_reduction(audio_array, sample_rate, **kwargs)
            else:
                self.logger.warning(f"Unknown method: {self.method}, returning original")
                return audio_array
                
        except Exception as e:
            self.logger.error(f"Noise reduction failed: {str(e)}")
            return audio_array  # Return original if reduction fails
    
    async def _spectral_subtraction(
        self, 
        audio_array: np.ndarray, 
        sample_rate: int,
        noise_estimate_duration: float = 1.0,
        alpha: float = 2.0,
        beta: float = 0.01
    ) -> np.ndarray:
        """
        Apply spectral subtraction noise reduction.
        
        Args:
            audio_array: Input audio array
            sample_rate: Sample rate
            noise_estimate_duration: Duration to use for noise estimation
            alpha: Over-subtraction factor
            beta: Spectral floor
            
        Returns:
            Noise-reduced audio array
        """
        try:
            self.logger.info("Applying spectral subtraction")
            
            # Estimate noise from the beginning of the audio
            noise_samples = int(noise_estimate_duration * sample_rate)
            noise_estimate = audio_array[:noise_samples]
            
            # Compute noise spectrum (should be array, not scalar)
            noise_spectrum = np.abs(np.fft.fft(noise_estimate))
            
            # Process audio in frames
            frame_length = 1024
            hop_length = frame_length // 2
            
            # Pad audio to ensure complete frames
            padded_length = ((len(audio_array) - frame_length) // hop_length + 1) * hop_length + frame_length
            padded_audio = np.pad(audio_array, (0, padded_length - len(audio_array)))
            
            processed_frames = []
            
            for i in range(0, len(padded_audio) - frame_length + 1, hop_length):
                frame = padded_audio[i:i + frame_length]
                
                # Apply window function
                window = np.hanning(frame_length)
                frame = frame * window
                
                # Compute spectrum
                frame_spectrum = np.fft.fft(frame)
                magnitude = np.abs(frame_spectrum)
                phase = np.angle(frame_spectrum)
                
                 # Apply spectral subtraction (CRITICAL FIX for indexing issue)
                try:
                    # Ensure noise_spectrum and magnitude have compatible lengths
                    if len(noise_spectrum) >= len(magnitude):
                        noise_threshold = alpha * noise_spectrum[:len(magnitude)]
                    else:
                        # Pad noise spectrum if shorter than magnitude
                        noise_threshold = alpha * np.pad(noise_spectrum, (0, len(magnitude) - len(noise_spectrum)), mode='edge')
                    
                    # Validate noise_threshold is not scalar
                    if np.isscalar(noise_threshold):
                        self.logger.warning("Noise threshold is scalar, converting to array")
                        noise_threshold = np.full_like(magnitude, noise_threshold)
                        
                except Exception as indexing_error:
                    self.logger.error(f"Indexing error in spectral subtraction: {str(indexing_error)}")
                    # Fallback: use simple noise reduction
                    noise_threshold = alpha * np.mean(noise_spectrum) * np.ones_like(magnitude)
                
                magnitude_subtracted = magnitude - noise_threshold
                magnitude_subtracted = np.maximum(magnitude_subtracted, beta * magnitude)
                
                # Reconstruct signal
                frame_spectrum_processed = magnitude_subtracted * np.exp(1j * phase)
                frame_processed = np.real(np.fft.ifft(frame_spectrum_processed))
                
                # Apply inverse window
                frame_processed = frame_processed * window
                
                processed_frames.append(frame_processed)
            
            # Overlap-add reconstruction
            output_length = len(processed_frames) * hop_length + frame_length
            output = np.zeros(output_length)
            
            for i, frame in enumerate(processed_frames):
                start = i * hop_length
                end = start + frame_length
                output[start:end] += frame
            
            # Normalize and trim to original length
            output = output[:len(audio_array)]
            output = output / np.max(np.abs(output)) * np.max(np.abs(audio_array))
            
            self.logger.info("Spectral subtraction completed")
            return output
            
        except Exception as e:
            self.logger.error(f"Spectral subtraction failed: {str(e)}")
            self.logger.warning("Returning original audio without noise reduction")
            return audio_array
    
    async def _wiener_filtering(
        self, 
        audio_array: np.ndarray, 
        sample_rate: int,
        noise_estimate_duration: float = 1.0,
        frame_length: int = 1024
    ) -> np.ndarray:
        """
        Apply Wiener filtering for noise reduction.
        
        Args:
            audio_array: Input audio array
            sample_rate: Sample rate
            noise_estimate_duration: Duration to use for noise estimation
            frame_length: Length of processing frames
            
        Returns:
            Noise-reduced audio array
        """
        try:
            self.logger.info("Applying Wiener filtering")
            
            # Estimate noise from the beginning
            noise_samples = int(noise_estimate_duration * sample_rate)
            noise_estimate = audio_array[:noise_samples]
            
            # Compute noise power spectrum
            noise_spectrum = np.mean(np.abs(np.fft.fft(noise_estimate)) ** 2, axis=0)
            
            # Process in frames
            hop_length = frame_length // 2
            processed_frames = []
            
            for i in range(0, len(audio_array) - frame_length + 1, hop_length):
                frame = audio_array[i:i + frame_length]
                
                # Apply window
                window = np.hanning(frame_length)
                frame = frame * window
                
                # Compute spectrum
                frame_spectrum = np.fft.fft(frame)
                signal_power = np.abs(frame_spectrum) ** 2
                
                # Wiener filter
                noise_power = noise_spectrum[:len(signal_power)]
                snr = signal_power / (noise_power + 1e-10)
                wiener_gain = snr / (snr + 1)
                
                # Apply filter
                frame_spectrum_filtered = frame_spectrum * wiener_gain
                frame_filtered = np.real(np.fft.ifft(frame_spectrum_filtered))
                
                # Apply inverse window
                frame_filtered = frame_filtered * window
                
                processed_frames.append(frame_filtered)
            
            # Overlap-add reconstruction
            output_length = len(processed_frames) * hop_length + frame_length
            output = np.zeros(output_length)
            
            for i, frame in enumerate(processed_frames):
                start = i * hop_length
                end = start + frame_length
                output[start:end] += frame
            
            # Normalize and trim
            output = output[:len(audio_array)]
            output = output / np.max(np.abs(output)) * np.max(np.abs(audio_array))
            
            self.logger.info("Wiener filtering completed")
            return output
            
        except Exception as e:
            self.logger.error(f"Wiener filtering failed: {str(e)}")
            return audio_array
    
    async def _voice_activity_detection(
        self, 
        audio_array: np.ndarray, 
        sample_rate: int,
        frame_length: int = 1024,
        energy_threshold: float = 0.1
    ) -> np.ndarray:
        """
        Apply Voice Activity Detection for noise reduction.
        
        Args:
            audio_array: Input audio array
            sample_rate: Sample rate
            frame_length: Length of processing frames
            energy_threshold: Energy threshold for voice detection
            
        Returns:
            Noise-reduced audio array
        """
        try:
            self.logger.info("Applying Voice Activity Detection")
            
            hop_length = frame_length // 2
            voice_mask = np.zeros(len(audio_array), dtype=bool)
            
            # Process in frames to detect voice activity
            for i in range(0, len(audio_array) - frame_length + 1, hop_length):
                frame = audio_array[i:i + frame_length]
                
                # Compute frame energy
                frame_energy = np.mean(frame ** 2)
                
                # Apply threshold
                if frame_energy > energy_threshold:
                    voice_mask[i:i + frame_length] = True
            
            # Apply mask to audio
            output = audio_array.copy()
            output[~voice_mask] *= 0.1  # Reduce noise in non-voice regions
            
            self.logger.info("Voice Activity Detection completed")
            return output
            
        except Exception as e:
            self.logger.error(f"Voice Activity Detection failed: {str(e)}")
            return audio_array
    
    async def _adaptive_noise_reduction(
        self, 
        audio_array: np.ndarray, 
        sample_rate: int,
        adaptation_rate: float = 0.01
    ) -> np.ndarray:
        """
        Apply adaptive noise reduction.
        
        Args:
            audio_array: Input audio array
            sample_rate: Sample rate
            adaptation_rate: Rate of adaptation
            
        Returns:
            Noise-reduced audio array
        """
        try:
            self.logger.info("Applying adaptive noise reduction")
            
            # Simple adaptive approach
            # In a real implementation, this would use more sophisticated algorithms
            
            # Estimate noise level adaptively
            noise_level = np.mean(np.abs(audio_array[:sample_rate]))  # First second
            
            # Apply adaptive thresholding
            threshold = noise_level * 2.0
            output = audio_array.copy()
            
            # Reduce samples below threshold
            mask = np.abs(output) < threshold
            output[mask] *= 0.5
            
            self.logger.info("Adaptive noise reduction completed")
            return output
            
        except Exception as e:
            self.logger.error(f"Adaptive noise reduction failed: {str(e)}")
            return audio_array
    
    async def _combined_reduction(
        self, 
        audio_array: np.ndarray, 
        sample_rate: int
    ) -> np.ndarray:
        """
        Apply combined noise reduction approach.
        
        Args:
            audio_array: Input audio array
            sample_rate: Sample rate
            
        Returns:
            Noise-reduced audio array
        """
        try:
            self.logger.info("Applying combined noise reduction")
            
            # Apply multiple methods in sequence
            output = audio_array.copy()
            
            # Step 1: Spectral subtraction
            output = await self._spectral_subtraction(output, sample_rate)
            
            # Step 2: Wiener filtering
            output = await self._wiener_filtering(output, sample_rate)
            
            # Step 3: Voice Activity Detection
            output = await self._voice_activity_detection(output, sample_rate)
            
            self.logger.info("Combined noise reduction completed")
            return output
            
        except Exception as e:
            self.logger.error(f"Combined reduction failed: {str(e)}")
            return audio_array
    
    def get_method_info(self) -> Dict[str, Any]:
        """Get information about the current noise reduction method."""
        return {
            "method": self.method,
            "supported_methods": self.supported_methods,
            "description": self._get_method_description()
        }
    
    def _get_method_description(self) -> str:
        """Get description of the current method."""
        descriptions = {
            "spectral": "Spectral subtraction - removes stationary noise",
            "wiener": "Wiener filtering - optimal linear filtering",
            "vad": "Voice Activity Detection - preserves voice regions",
            "adaptive": "Adaptive noise reduction - adjusts to signal",
            "combined": "Combined approach - multiple methods in sequence"
        }
        return descriptions.get(self.method, "Unknown method")
    
    def set_method(self, method: str):
        """Change the noise reduction method."""
        if method in self.supported_methods:
            self.method = method
            self.logger.info(f"Noise reduction method changed to: {method}")
        else:
            self.logger.warning(f"Unsupported method: {method}")


# Example usage and testing
async def main():
    """Example usage of NoiseReducer."""
    # Create test audio with noise
    sample_rate = 16000
    duration = 5.0  # 5 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Clean signal (sine wave)
    clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    # Add noise
    noise = 0.1 * np.random.randn(len(t))
    noisy_signal = clean_signal + noise
    
    print(f"Original signal SNR: {np.var(clean_signal) / np.var(noise):.2f}")
    
    # Test different noise reduction methods
    methods = ["spectral", "wiener", "vad", "adaptive", "combined"]
    
    for method in methods:
        reducer = NoiseReducer(reduction_method=method)
        reduced_signal = await reducer.reduce_noise(noisy_signal, sample_rate)
        
        # Calculate SNR improvement
        residual_noise = reduced_signal - clean_signal
        snr_improvement = np.var(clean_signal) / np.var(residual_noise)
        
        print(f"{method.capitalize()}: SNR = {snr_improvement:.2f}")


if __name__ == "__main__":
    # Run example
    asyncio.run(main())
