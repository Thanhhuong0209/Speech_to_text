import streamlit as st
from backend import get_backend
import tempfile
import os

st.set_page_config(
    page_title="Simple Speech-to-Text",
    page_icon="",
    layout="wide"
)

st.title("🎤 Simple Speech-to-Text with Whisper")
st.markdown("Upload audio file and get transcription instantly!")

# Initialize backend
backend = get_backend()

# File upload
uploaded_file = st.file_uploader(
    "Choose an audio file", 
    type=['wav', 'mp3', 'flac', 'm4a']
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Transcribe audio
        if st.button(" Transcribe Audio", type="primary"):
            with st.spinner("Processing audio..."):
                result = backend.transcribe_audio(tmp_file_path)
                
                if result:
                    st.success("✅ Transcription completed!")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📝 Transcribed Text")
                        st.write(result['text'])
                        
                        if result.get('language'):
                            st.info(f" Detected Language: {result['language']}")
                    
                    with col2:
                        st.subheader("📊 Audio Info")
                        st.write(f"**File:** {uploaded_file.name}")
                        st.write(f"**Size:** {uploaded_file.size} bytes")
                        st.write(f"**Type:** {uploaded_file.type}")
                        
                        if result.get('segments'):
                            st.write(f"**Segments:** {len(result['segments'])}")
                    
                    # Download transcription
                    st.download_button(
                        label="📥 Download Transcription",
                        data=result['text'],
                        file_name="transcription.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("❌ Transcription failed!")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Simple Speech-to-Text App**
    
    - Uses OpenAI Whisper model
    - Supports multiple audio formats
    - Fast and accurate transcription
    - No complex backend dependencies
    """)
    
    st.header("🔧 Settings")
    model_size = st.selectbox(
        "Whisper Model Size",
        ["tiny", "base", "small", "medium", "large"],
        index=1
    )
    
    if st.button("🔄 Change Model"):
        st.info(f"Model will be changed to {model_size} on next restart")
        backend.model_name = model_size
        backend.model = None  # Force reload

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Whisper")
