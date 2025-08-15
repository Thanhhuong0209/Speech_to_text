import streamlit as st
import numpy as np
import whisper
import tempfile
import os
from pathlib import Path

class SimpleBackend:
    def __init__(self):
        self.model = None
        self.model_name = "base"
    
    def load_model(self):
        """Load Whisper model"""
        try:
            if self.model is None:
                st.info(f"Loading Whisper {self.model_name} model...")
                self.model = whisper.load_model(self.model_name)
                st.success(f"Whisper {self.model_name} model loaded successfully!")
            return self.model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    def transcribe_audio(self, audio_file):
        """Transcribe audio file"""
        try:
            model = self.load_model()
            if model is None:
                return None
            
            st.info("Transcribing audio...")
            result = model.transcribe(audio_file)
            
            return {
                'text': result['text'],
                'language': result.get('language', 'en'),
                'segments': result.get('segments', [])
            }
        except Exception as e:
            st.error(f"Transcription error: {str(e)}")
            return None

# Global backend instance
@st.cache_resource
def get_backend():
    return SimpleBackend()
