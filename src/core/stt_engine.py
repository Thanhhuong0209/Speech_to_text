"""
Main Speech-to-Text Engine

This module provides the core STT functionality that orchestrates audio processing,
noise reduction, transcription, and post-processing to deliver high-quality results.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

from ..models.whisper_wrapper import WhisperWrapper
from ..core.audio_processor import AudioProcessor
from ..core.noise_reducer import NoiseReducer
from ..core.ai_insights_engine import AIInsightsEngine
from ..utils.config import Config
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class TranscriptionResult:
    """Result of transcription process with metadata."""
    text: str
    confidence: float
    language: str
    duration: float
    word_timestamps: List[Tuple[str, float, float]]
    processing_time: float
    audio_quality_score: float
    noise_reduction_applied: bool


class STTEngine:
    """
    Main Speech-to-Text engine that orchestrates the entire transcription pipeline.
    
    This engine provides a unified interface for audio transcription with features
    including noise reduction, audio enhancement, and post-processing optimization.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the STT engine with configuration.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        self.logger = logger
        
        # Initialize components
        self.whisper_model = WhisperWrapper(
            model_size=self.config.whisper_model_size
        )
        self.audio_processor = AudioProcessor(
            sample_rate=self.config.sample_rate,
            chunk_size=self.config.audio_chunk_size
        )
        self.noise_reducer = NoiseReducer(
            reduction_method=self.config.noise_reduction_method
        )
        self.ai_insights = AIInsightsEngine(config)
        
        self.logger.info(f"STT Engine initialized with {self.config.whisper_model_size} model and AI insights")
    
    async def transcribe_audio(
        self,
        audio_data: Union[str, Path, bytes, np.ndarray],
        language: Optional[str] = None,
        apply_noise_reduction: bool = True,
        enhance_audio: bool = True,
        post_process: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe audio data with comprehensive processing pipeline.
        
        Args:
            audio_data: Audio input (file path, bytes, or numpy array)
            language: Expected language code (auto-detected if None)
            apply_noise_reduction: Whether to apply noise reduction
            enhance_audio: Whether to enhance audio quality
            post_process: Whether to apply post-processing optimization
            
        Returns:
            TranscriptionResult with transcription and metadata
            
        Raises:
            ValueError: If audio data is invalid
            RuntimeError: If transcription fails
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # CRITICAL: Ensure Whisper model is loaded before transcription
            await self.whisper_model.load_model()
            
            # Validate input
            if audio_data is None:
                raise ValueError("Audio data cannot be None")
            
            self.logger.info(f"Starting transcription for audio: {type(audio_data)}")
            
            # CRITICAL: Load Whisper model before transcription
            await self.whisper_model.load_model()
            self.logger.info("Whisper model loaded successfully")
            
            # Step 1: Load and preprocess audio
            audio_array, sample_rate = await self.audio_processor.load_audio(audio_data)
            duration = len(audio_array) / sample_rate
            
            self.logger.info(f"Audio loaded: {duration:.2f}s, {sample_rate}Hz")
            
            # Step 2: Audio enhancement (optional)
            if enhance_audio:
                audio_array = await self.audio_processor.enhance_audio(audio_array)
                self.logger.info("Audio enhancement applied")
            
            # Step 3: Noise reduction (optional)
            if apply_noise_reduction:
                audio_array = await self.noise_reducer.reduce_noise(audio_array, sample_rate)
                self.logger.info("Noise reduction applied")
            
            # Step 4: Transcription with Whisper
            transcription = await self.whisper_model.transcribe(
                audio_array, 
                sample_rate, 
                language=language
            )
            
            # Step 5: Post-processing (optional)
            if post_process:
                transcription = await self._post_process_transcription(transcription)
                self.logger.info("Post-processing applied")
            
            # Step 6: Calculate confidence and quality metrics
            confidence = self._calculate_confidence(transcription)
            quality_score = self._calculate_audio_quality(audio_array, sample_rate)
            
            # Step 7: Extract word timestamps
            word_timestamps = self._extract_word_timestamps(transcription)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            result = TranscriptionResult(
                text=transcription["text"],
                confidence=confidence,
                language=transcription.get("language", "unknown"),
                duration=duration,
                word_timestamps=word_timestamps,
                processing_time=processing_time,
                audio_quality_score=quality_score,
                noise_reduction_applied=apply_noise_reduction
            )
            
            self.logger.info(
                f"Transcription completed in {processing_time:.2f}s. "
                f"Confidence: {confidence:.2f}, Quality: {quality_score:.2f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise RuntimeError(f"Transcription failed: {str(e)}") from e
    
    async def transcribe_stream(
        self,
        audio_stream,
        chunk_duration: float = 10.0,
        overlap: float = 1.0
    ) -> List[TranscriptionResult]:
        """
        Transcribe audio stream in real-time chunks.
        
        Args:
            audio_stream: Streaming audio source
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
            
        Returns:
            List of TranscriptionResult for each chunk
        """
        results = []
        chunk_size = int(chunk_duration * self.config.sample_rate)
        overlap_size = int(overlap * self.config.sample_rate)
        
        self.logger.info(f"Starting streaming transcription: {chunk_duration}s chunks")
        
        try:
            buffer = np.array([], dtype=np.float32)
            
            async for chunk in audio_stream:
                buffer = np.concatenate([buffer, chunk])
                
                while len(buffer) >= chunk_size:
                    # Extract chunk with overlap
                    chunk_data = buffer[:chunk_size]
                    buffer = buffer[chunk_size - overlap_size:]
                    
                    # Transcribe chunk
                    result = await self.transcribe_audio(
                        chunk_data,
                        apply_noise_reduction=True,
                        enhance_audio=True,
                        post_process=True
                    )
                    
                    results.append(result)
                    
                    self.logger.info(f"Chunk transcribed: {result.text[:50]}...")
            
            # Process remaining buffer
            if len(buffer) > 0:
                result = await self.transcribe_audio(
                    buffer,
                    apply_noise_reduction=True,
                    enhance_audio=True,
                    post_process=True
                )
                results.append(result)
            
            self.logger.info(f"Streaming transcription completed: {len(results)} chunks")
            return results
            
        except Exception as e:
            self.logger.error(f"Streaming transcription failed: {str(e)}")
            raise RuntimeError(f"Streaming transcription failed: {str(e)}") from e
    
    async def transcribe_with_insights(
        self,
        audio_data: Union[str, Path, bytes, np.ndarray],
        analysis_type: str = "summary",
        language: Optional[str] = None,
        apply_noise_reduction: bool = True,
        enhance_audio: bool = True,
        post_process: bool = True,
        custom_prompt: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict:
        """
        Transcribe audio and generate AI-powered insights in one operation.
        
        Args:
            audio_data: Audio input (file path, bytes, or numpy array)
            analysis_type: Type of AI analysis (summary, qa, insights, sentiment, actions, topics)
            language: Expected language code (auto-detected if None)
            apply_noise_reduction: Whether to apply noise reduction
            enhance_audio: Whether to enhance audio quality
            post_process: Whether to apply post-processing optimization
            custom_prompt: Custom prompt for AI analysis
            context: Additional context for AI analysis
            
        Returns:
            Dictionary containing transcription and AI insights
        """
        try:
            self.logger.info(f"Starting speech-to-insights processing with {analysis_type} analysis")
            
            # First, transcribe the audio
            transcription_result = await self.transcribe_audio(
                audio_data=audio_data,
                language=language,
                apply_noise_reduction=apply_noise_reduction,
                enhance_audio=enhance_audio,
                post_process=post_process
            )
            
            # Then, generate AI insights from the transcription
            insights_result = await self.ai_insights.analyze_transcription(
                transcription_text=transcription_result.text,
                analysis_type=analysis_type,
                custom_prompt=custom_prompt,
                context=context
            )
            
            # Combine results
            combined_result = {
                "transcription": {
                    "text": transcription_result.text,
                    "confidence": transcription_result.confidence,
                    "language": transcription_result.language,
                    "duration": transcription_result.duration,
                    "processing_time": transcription_result.processing_time,
                    "audio_quality_score": transcription_result.audio_quality_score,
                    "noise_reduction_applied": transcription_result.noise_reduction_applied,
                    "word_timestamps": transcription_result.word_timestamps
                },
                "ai_insights": insights_result,
                "total_processing_time": transcription_result.processing_time + insights_result.get("processing_time", 0),
                "status": "completed"
            }
            
            self.logger.info(f"Speech-to-insights processing completed successfully")
            return combined_result
            
        except Exception as e:
            self.logger.error(f"Speech-to-insights processing failed: {str(e)}")
            raise RuntimeError(f"Speech-to-insights failed: {str(e)}")
    
    async def batch_transcribe_with_insights(
        self,
        audio_files: List[Union[str, Path, bytes, np.ndarray]],
        analysis_type: str = "summary",
        language: Optional[str] = None,
        apply_noise_reduction: bool = True,
        enhance_audio: bool = True,
        post_process: bool = True
    ) -> List[Dict]:
        """
        Process multiple audio files with transcription and insights generation.
        
        Args:
            audio_files: List of audio inputs
            analysis_type: Type of AI analysis to perform
            language: Expected language code
            apply_noise_reduction: Whether to apply noise reduction
            enhance_audio: Whether to enhance audio quality
            post_process: Whether to apply post-processing
            
        Returns:
            List of results with transcription and insights
        """
        try:
            self.logger.info(f"Starting batch speech-to-insights processing for {len(audio_files)} files")
            
            results = []
            for i, audio_file in enumerate(audio_files):
                try:
                    result = await self.transcribe_with_insights(
                        audio_data=audio_file,
                        analysis_type=analysis_type,
                        language=language,
                        apply_noise_reduction=apply_noise_reduction,
                        enhance_audio=enhance_audio,
                        post_process=post_process
                    )
                    result["file_index"] = i
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process file {i}: {str(e)}")
                    results.append({
                        "file_index": i,
                        "error": str(e),
                        "status": "failed"
                    })
                
                # Add small delay between files
                await asyncio.sleep(0.1)
            
            # Generate summary of all analyses
            if results:
                insights_results = [r.get("ai_insights", {}) for r in results if r.get("status") == "completed"]
                if insights_results:
                    summary = await self.ai_insights.get_analysis_summary(insights_results)
                    combined_result = {
                        "individual_results": results,
                        "summary": summary,
                        "total_files": len(audio_files),
                        "successful_files": len([r for r in results if r.get("status") == "completed"]),
                        "status": "completed"
                    }
                    return combined_result
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch speech-to-insights processing failed: {str(e)}")
            raise RuntimeError(f"Batch processing failed: {str(e)}")
    
    async def _post_process_transcription(self, transcription: Dict) -> Dict:
        """
        Apply post-processing to improve transcription quality.
        
        Args:
            transcription: Raw transcription result
            
        Returns:
            Enhanced transcription result
        """
        try:
            text = transcription["text"]
            
            # Basic text cleaning
            text = text.strip()
            text = " ".join(text.split())  # Remove extra whitespace
            
            # Capitalization fixes
            text = text.capitalize()
            
            # Punctuation improvements
            if not text.endswith(('.', '!', '?')):
                text += '.'
            
            transcription["text"] = text
            return transcription
            
        except Exception as e:
            self.logger.warning(f"Post-processing failed: {str(e)}")
            return transcription
    
    def _calculate_confidence(self, transcription: Dict) -> float:
        """
        Calculate confidence score for transcription.
        
        Args:
            transcription: Transcription result
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # For now, return a default confidence
            # In a real implementation, this would use Whisper's confidence scores
            return 0.85
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {str(e)}")
            return 0.5
    
    def _calculate_audio_quality(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Calculate audio quality score.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            # Calculate signal-to-noise ratio
            signal_power = np.mean(audio ** 2)
            noise_power = np.var(audio)
            
            if noise_power == 0:
                return 1.0
            
            snr = 10 * np.log10(signal_power / noise_power)
            
            # Normalize to 0-1 range
            quality = min(1.0, max(0.0, (snr + 20) / 40))
            
            return quality
            
        except Exception as e:
            self.logger.warning(f"Quality calculation failed: {str(e)}")
            return 0.5
    
    def _extract_word_timestamps(self, transcription: Dict) -> List[Tuple[str, float, float]]:
        """
        Extract word-level timestamps from transcription.
        
        Args:
            transcription: Transcription result
            
        Returns:
            List of (word, start_time, end_time) tuples
        """
        try:
            # For now, return basic word segmentation
            # In a real implementation, this would use Whisper's timestamp data
            words = transcription["text"].split()
            duration = len(words) * 0.5  # Rough estimate
            
            timestamps = []
            for i, word in enumerate(words):
                start_time = i * duration / len(words)
                end_time = (i + 1) * duration / len(words)
                timestamps.append((word, start_time, end_time))
            
            return timestamps
            
        except Exception as e:
            self.logger.warning(f"Timestamp extraction failed: {str(e)}")
            return []
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_size": self.config.whisper_model_size,
            "sample_rate": self.config.sample_rate,
            "chunk_size": self.config.audio_chunk_size,
            "noise_reduction": self.config.noise_reduction_method,
            "supported_languages": self.whisper_model.get_supported_languages()
        }
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            await self.whisper_model.cleanup()
            self.logger.info("STT Engine cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")


# Assertions for testing
if __name__ == "__main__":
    # Test basic functionality
    engine = STTEngine()
    assert engine is not None
    assert engine.whisper_model is not None
    assert engine.audio_processor is not None
    assert engine.noise_reducer is not None
    print("STT Engine initialization test passed!")
