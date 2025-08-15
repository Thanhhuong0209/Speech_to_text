"""
Enhanced AI Insights Engine with GPT-4-mini integration
Provides intelligent analysis of transcribed audio content
"""

import logging
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from ..utils.logging import setup_logger
from ..utils.config import get_config

logger = setup_logger(__name__)

@dataclass
class AudioInsights:
    """Structured audio insights data"""
    summary: str
    sentiment: Dict[str, float]
    topics: List[Dict[str, Any]]
    speakers: List[Dict[str, Any]]
    action_items: List[str]
    language: str
    duration: float
    word_count: int
    confidence: float
    processing_time: float

@dataclass
class SpeakerInfo:
    """Speaker detection information"""
    speaker_id: str
    gender: str
    duration_percentage: float
    word_count: int
    role: str
    confidence: float

class AIInsightsEngineV2:
    """
    Enhanced AI insights engine with GPT-4-mini integration
    Provides comprehensive audio content analysis
    """
    
    def __init__(self, config=None):
        """Initialize the enhanced AI insights engine"""
        self.logger = logger
        self.config = config or get_config()
        self.openai_api_key = self.config.get("openai_api_key")
        self.model = "gpt-4-1106-preview"  # GPT-4-mini for cost efficiency
        
        # Analysis prompts
        self.analysis_prompts = {
            "summary": "Provide a concise summary of the following audio transcript in 2-3 sentences:",
            "sentiment": "Analyze the sentiment of this transcript. Return JSON with positive, neutral, negative scores (0-1):",
            "topics": "Identify the main topics discussed in this transcript. Return JSON with topic names and confidence scores:",
            "speakers": "Analyze speaker patterns in this transcript. Return JSON with speaker count, gender distribution, and roles:",
            "action_items": "Extract action items and next steps from this transcript. Return as a numbered list:"
        }
        
        self.logger.info("Enhanced AI Insights Engine initialized with GPT-4-mini")
    
    async def analyze_transcript(self, 
                                transcript: str, 
                                audio_metadata: Dict[str, Any],
                                analysis_types: List[str] = None) -> AudioInsights:
        """
        Analyze transcribed audio content with AI insights
        
        Args:
            transcript: Transcribed text content
            audio_metadata: Audio file metadata (duration, format, etc.)
            analysis_types: Types of analysis to perform
            
        Returns:
            AudioInsights object with comprehensive analysis
        """
        if not analysis_types:
            analysis_types = ["summary", "sentiment", "topics", "speakers", "action_items"]
        
        self.logger.info(f"Starting AI analysis for transcript: {len(transcript)} characters")
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Perform parallel analysis
            analysis_tasks = []
            for analysis_type in analysis_types:
                if analysis_type in self.analysis_prompts:
                    task = self._analyze_content(transcript, analysis_type)
                    analysis_tasks.append(task)
            
            # Wait for all analysis to complete
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process results
            insights = self._process_analysis_results(analysis_results, analysis_types)
            
            # Add metadata
            insights.language = audio_metadata.get("language", "en")
            insights.duration = audio_metadata.get("duration", 0.0)
            insights.word_count = len(transcript.split())
            insights.confidence = audio_metadata.get("confidence", 0.0)
            insights.processing_time = asyncio.get_event_loop().time() - start_time
            
            self.logger.info(f"AI analysis completed in {insights.processing_time:.2f}s")
            return insights
            
        except Exception as e:
            self.logger.error(f"AI analysis failed: {str(e)}")
            # Return fallback insights
            return self._create_fallback_insights(transcript, audio_metadata)
    
    async def _analyze_content(self, transcript: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze content using OpenAI API"""
        try:
            if not self.openai_api_key:
                return self._local_analysis(transcript, analysis_type)
            
            prompt = self.analysis_prompts[analysis_type]
            full_prompt = f"{prompt}\n\nTranscript: {transcript}"
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert audio content analyst."},
                        {"role": "user", "content": full_prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.3
                }
                
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]
                        return self._parse_ai_response(content, analysis_type)
                    else:
                        self.logger.warning(f"OpenAI API failed: {response.status}")
                        return self._local_analysis(transcript, analysis_type)
                        
        except Exception as e:
            self.logger.error(f"Content analysis failed for {analysis_type}: {str(e)}")
            return self._local_analysis(transcript, analysis_type)
    
    def _local_analysis(self, transcript: str, analysis_type: str) -> Dict[str, Any]:
        """Fallback local analysis when AI API is unavailable"""
        if analysis_type == "summary":
            words = transcript.split()
            if len(words) > 20:
                summary = " ".join(words[:20]) + "..."
            else:
                summary = transcript
            return {"summary": summary}
        
        elif analysis_type == "sentiment":
            # Simple keyword-based sentiment
            positive_words = ["good", "great", "excellent", "amazing", "wonderful", "happy"]
            negative_words = ["bad", "terrible", "awful", "horrible", "sad", "angry"]
            
            text_lower = transcript.lower()
            positive_score = sum(1 for word in positive_words if word in text_lower) / len(positive_words)
            negative_score = sum(1 for word in negative_words if word in text_lower) / len(negative_words)
            neutral_score = 1 - (positive_score + negative_score)
            
            return {
                "sentiment": {
                    "positive": min(positive_score, 1.0),
                    "neutral": max(neutral_score, 0.0),
                    "negative": min(negative_score, 1.0)
                }
            }
        
        elif analysis_type == "topics":
            # Extract common words as topics
            words = transcript.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 4 and word.isalpha():
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top 5 topics
            topics = []
            for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]:
                topics.append({
                    "topic": word.capitalize(),
                    "confidence": min(freq / len(words), 1.0),
                    "mentions": freq
                })
            
            return {"topics": topics}
        
        elif analysis_type == "speakers":
            # Simple speaker estimation
            sentences = transcript.split('.')
            estimated_speakers = min(len(sentences), 3)
            
            speakers = []
            for i in range(estimated_speakers):
                speakers.append({
                    "speaker_id": f"Speaker {chr(65+i)}",
                    "gender": "Unknown",
                    "duration_percentage": 100.0 / estimated_speakers,
                    "word_count": len(transcript.split()) // estimated_speakers,
                    "role": "Participant",
                    "confidence": 0.7
                })
            
            return {"speakers": speakers}
        
        elif analysis_type == "action_items":
            # Extract action words
            action_words = ["will", "should", "must", "need to", "going to", "plan to"]
            sentences = transcript.split('.')
            action_items = []
            
            for sentence in sentences:
                if any(action in sentence.lower() for action in action_words):
                    action_items.append(sentence.strip())
            
            if not action_items:
                action_items = ["No specific action items identified"]
            
            return {"action_items": action_items[:5]}
        
        return {}
    
    def _parse_ai_response(self, content: str, analysis_type: str) -> Dict[str, Any]:
        """Parse AI API response into structured data"""
        try:
            if analysis_type == "sentiment":
                # Try to extract JSON from response
                if "{" in content and "}" in content:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    json_str = content[start:end]
                    return json.loads(json_str)
                else:
                    return self._local_analysis("", analysis_type)
            
            elif analysis_type == "topics":
                if "{" in content and "}" in content:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    json_str = content[start:end]
                    return json.loads(json_str)
                else:
                    return self._local_analysis("", analysis_type)
            
            elif analysis_type == "speakers":
                if "{" in content and "}" in content:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    json_str = content[start:end]
                    return json.loads(json_str)
                else:
                    return self._local_analysis("", analysis_type)
            
            elif analysis_type == "action_items":
                # Extract numbered or bulleted items
                lines = content.split('\n')
                items = []
                for line in lines:
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith(('•', '-', '*'))):
                        items.append(line)
                
                if items:
                    return {"action_items": items}
                else:
                    return {"action_items": [content.strip()]}
            
            else:
                return {analysis_type: content.strip()}
                
        except Exception as e:
            self.logger.error(f"Failed to parse AI response for {analysis_type}: {str(e)}")
            return self._local_analysis("", analysis_type)
    
    def _process_analysis_results(self, results: List[Any], analysis_types: List[str]) -> AudioInsights:
        """Process analysis results into AudioInsights object"""
        insights_data = {}
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Analysis {analysis_types[i]} failed: {str(result)}")
                continue
            
            if isinstance(result, dict):
                insights_data.update(result)
        
        # Create AudioInsights object with defaults
        return AudioInsights(
            summary=insights_data.get("summary", "Analysis not available"),
            sentiment=insights_data.get("sentiment", {"positive": 0.5, "neutral": 0.3, "negative": 0.2}),
            topics=insights_data.get("topics", []),
            speakers=insights_data.get("speakers", []),
            action_items=insights_data.get("action_items", []),
            language="en",
            duration=0.0,
            word_count=0,
            confidence=0.0,
            processing_time=0.0
        )
    
    def _create_fallback_insights(self, transcript: str, audio_metadata: Dict[str, Any]) -> AudioInsights:
        """Create fallback insights when analysis fails"""
        return AudioInsights(
            summary="Analysis temporarily unavailable. Please try again later.",
            sentiment={"positive": 0.5, "neutral": 0.3, "negative": 0.2},
            topics=[],
            speakers=[],
            action_items=[],
            language=audio_metadata.get("language", "en"),
            duration=audio_metadata.get("duration", 0.0),
            word_count=len(transcript.split()),
            confidence=audio_metadata.get("confidence", 0.0),
            processing_time=0.0
        )
    
    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Enhanced AI Insights Engine cleanup completed")
