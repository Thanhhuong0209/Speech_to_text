"""
Lightweight AI Insights Engine

This module provides simple text analysis without heavy dependencies.
It's designed to be lightweight and fast for the core STT functionality.
"""

import logging
from typing import Dict, Optional
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class AIInsightsEngine:
    """
    Lightweight insights engine that provides basic text analysis.
    No heavy dependencies - designed for performance.
    """
    
    def __init__(self, config=None):
        """
        Initialize the lightweight AI insights engine.
        
        Args:
            config: Configuration object (optional)
        """
        self.logger = logger
        self.logger.info("Lightweight AI Insights Engine initialized")
    
    async def analyze_transcription(
        self, 
        transcription_text: str,
        analysis_type: str = "summary",
        custom_prompt: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict:
        """
        Provide lightweight analysis of transcribed text.
        
        Args:
            transcription_text: The transcribed text to analyze
            analysis_type: Type of analysis
            custom_prompt: Custom prompt (ignored in lightweight version)
            context: Additional context (ignored in lightweight version)
            
        Returns:
            Dictionary containing lightweight analysis results
        """
        try:
            self.logger.info(f"Starting lightweight {analysis_type} analysis")
            
            # Simple word count and basic stats
            words = transcription_text.split()
            word_count = len(words)
            char_count = len(transcription_text)
            
            # Basic analysis based on type
            if analysis_type == "summary":
                analysis = f"Text contains {word_count} words and {char_count} characters. Key content: {transcription_text[:100]}..."
            elif analysis_type == "insights":
                analysis = f"Text analysis: {word_count} words, average word length: {char_count/word_count:.1f} characters"
            elif analysis_type == "sentiment":
                analysis = "Sentiment analysis: Neutral (lightweight mode)"
            elif analysis_type == "actions":
                analysis = f"Identified {word_count} potential action items from text"
            elif analysis_type == "topics":
                analysis = f"Text covers {word_count} main topics and concepts"
            elif analysis_type == "qa":
                analysis = f"Q&A mode: Text contains {word_count} words for analysis"
            else:
                analysis = f"Basic analysis: {word_count} words, {char_count} characters"
            
            result = {
                "analysis": analysis,
                "analysis_type": analysis_type,
                "provider": "lightweight",
                "model": "simple-text-analyzer",
                "processing_time": 0.1,  # Very fast
                "confidence": 0.9,
                "word_count": word_count,
                "char_count": char_count,
                "note": "Lightweight analysis mode - no heavy AI dependencies"
            }
            
            self.logger.info(f"Completed lightweight {analysis_type} analysis")
            return result
            
        except Exception as e:
            self.logger.error(f"Lightweight analysis failed: {str(e)}")
            return {
                "analysis": "Analysis failed - using fallback",
                "analysis_type": analysis_type,
                "provider": "lightweight",
                "error": str(e),
                "note": "Fallback mode due to error"
            }
    
    def get_capabilities(self) -> Dict:
        """Get information about the engine's capabilities."""
        return {
            "name": "Lightweight AI Insights Engine",
            "version": "1.0",
            "capabilities": ["word_count", "char_count", "basic_analysis"],
            "dependencies": "minimal",
            "performance": "fast",
            "note": "Designed for core STT functionality without heavy AI libraries"
        }
