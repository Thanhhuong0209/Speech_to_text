# Speech-to-Insights AI System

A comprehensive speech recognition and analysis system built with Streamlit, Whisper, and advanced audio processing capabilities. This system transforms audio into actionable insights through real-time transcription and AI-powered analysis.

## System Overview

The Speech-to-Insights AI System is designed to process audio files, perform speech-to-text transcription using OpenAI's Whisper model, and provide intelligent analysis including sentiment analysis, topic detection, and speaker identification. The system integrates with the LibriSpeech dataset for testing and validation purposes.

## Key Features

### Core Functionality
- **Audio Transcription**: Convert speech to text using Whisper models (tiny, base, small, medium)
- **Real-time Processing**: Streamlit-based web interface for immediate results
- **Multiple Input Methods**: Upload audio files, use LibriSpeech dataset, or record audio directly
- **Advanced Audio Processing**: Noise reduction, audio enhancement, and quality optimization

### AI Analysis Capabilities
- **Content Summary**: Automatic generation of audio content summaries
- **Sentiment Analysis**: Emotion detection and sentiment classification
- **Topic Detection**: Identify key themes and subjects discussed
- **Speaker Detection**: Multi-speaker identification and role analysis
- **Action Items Extraction**: Automatic identification of tasks and next steps

### Technical Features
- **SSL-Secure Connections**: Robust error handling for network operations
- **Asynchronous Processing**: Efficient handling of large audio files
- **Caching System**: Optimized performance with intelligent data caching
- **Modular Architecture**: Clean separation of concerns across components

## System Architecture

### Core Components

#### 1. Audio Processing Engine (`src/core/audio_processor.py`)
- Handles audio file loading and preprocessing
- Supports multiple audio formats (WAV, MP3, FLAC, M4A)
- Implements audio enhancement algorithms
- Manages sample rate conversion and normalization

#### 2. STT Engine (`src/core/stt_engine.py`)
- Integrates with Whisper models for transcription
- Manages model loading and caching
- Handles post-processing and confidence scoring
- Provides unified interface for different model sizes

#### 3. LibriSpeech Ingester (`src/data/librispeech_ingester.py`)
- Downloads and manages LibriSpeech dataset access
- Implements efficient tar.gz extraction
- Handles SSL connection management
- Provides real-time audio streaming capabilities

#### 4. Noise Reduction (`src/core/noise_reducer.py`)
- Spectral subtraction algorithms
- Adaptive noise filtering
- Quality improvement techniques
- Real-time audio enhancement

#### 5. AI Insights Engine (`src/core/ai_insights_engine.py`)
- Natural language processing for content analysis
- Sentiment analysis algorithms
- Topic modeling and classification
- Action item extraction

### Web Interface (`src/web/streamlit_app.py`)
- Clean, responsive Streamlit application
- Real-time processing status updates
- Interactive audio visualization
- Comprehensive results display

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Windows 10/11 (tested on Windows 10.0.26100)
- PowerShell or Command Prompt
- Internet connection for initial setup

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd BLUEBLEAP_AI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Dependencies

The system requires the following key packages:
- `streamlit`: Web application framework
- `openai-whisper`: Speech recognition models
- `aiohttp`: Asynchronous HTTP client/server
- `soundfile`: Audio file processing
- `numpy`: Numerical computing
- `plotly`: Data visualization
- `streamlit-webrtc`: Audio recording capabilities

## Usage Instructions

### Starting the Application

1. **Activate virtual environment**
   ```bash
   venv\Scripts\activate
   ```

2. **Launch Streamlit app**
   ```bash
   streamlit run src/web/streamlit_app.py
   ```

3. **Access the application**
   - Local URL: `http://localhost:8501`
   - Network URL: `http://<your-ip>:8501`

### Using the Application

#### Tab 1: Audio Transcription

**Upload Audio File**
1. Select "Upload Audio File" from input method options
2. Choose an audio file (WAV, MP3, FLAC, M4A)
3. Click "Transcribe Audio" to process
4. View transcription results with confidence metrics

**LibriSpeech Dataset**
1. Select "LibriSpeech Dataset" from input method options
2. Choose dataset subset (dev-clean, test-clean, train-clean-100, train-clean-360)
3. Set file index (0-999)
4. Click "Process LibriSpeech" to download and transcribe
5. Use "Random Sample" for different audio files

**Record Audio**
1. Select "Record Audio" from input method options
2. Click "Start" to begin recording
3. Speak into your microphone
4. Click "Stop Recording & Process" to analyze

#### Tab 2: AI Analysis

**Content Summary**
- Automatic generation of audio content summaries
- Key points and main themes identification
- Duration and participant information

**Sentiment Analysis**
- Overall sentiment classification (Positive/Negative/Neutral)
- Confidence scoring
- Emotion detection

**Topic Detection**
- Automatic topic identification
- Confidence levels for each topic
- Mention frequency analysis

**Speaker Analysis**
- Multi-speaker identification
- Speaking duration and word count
- Role and gender identification

**Action Items**
- Automatic extraction of tasks and next steps
- Prioritized action list
- Meeting follow-up items

### Configuration Options

#### Whisper Model Selection
- **Tiny**: Fastest processing, lower accuracy
- **Base**: Balanced speed and accuracy (recommended)
- **Small**: Higher accuracy, moderate speed
- **Medium**: Highest accuracy, slower processing

#### Performance Modes
- **Fast**: Optimized for speed, reduced quality
- **Balanced**: Best balance of speed and quality (default)
- **Accurate**: Maximum quality, slower processing

#### AI Analysis Features
- **Summary**: Content summarization
- **Sentiment**: Emotion and sentiment analysis
- **Topics**: Theme and subject detection
- **Speaker Detection**: Multi-speaker identification
- **Action Items**: Task extraction

## Technical Implementation Details

### Audio Processing Pipeline

1. **Audio Loading**
   - File format detection and validation
   - Sample rate conversion to 16kHz
   - Audio normalization and preprocessing

2. **Enhancement**
   - Spectral subtraction noise reduction
   - Audio quality improvement
   - Dynamic range optimization

3. **Transcription**
   - Whisper model loading and caching
   - Real-time audio processing
   - Confidence scoring and validation

4. **Post-processing**
   - Text cleaning and formatting
   - Language detection
   - Quality assessment

### Error Handling and Robustness

#### SSL Connection Management
- Automatic SSL context configuration
- Connection pooling and reuse
- Graceful error handling for network issues
- Retry mechanisms with exponential backoff

#### Resource Management
- Automatic cleanup of audio sessions
- Memory optimization for large files
- Cache management and garbage collection
- Graceful degradation on errors

#### Fallback Mechanisms
- Automatic fallback to sample data on errors
- Progressive error handling
- User-friendly error messages
- System stability maintenance

### Performance Optimizations

#### Caching Strategy
- Whisper model caching to avoid reloading
- Audio data caching for repeated processing
- Metadata caching for faster access
- Intelligent cache invalidation

#### Asynchronous Processing
- Non-blocking audio operations
- Concurrent download and processing
- Efficient resource utilization
- Responsive user interface

## Troubleshooting

### Common Issues

**SSL Connection Errors**
- Ensure stable internet connection
- Check firewall settings
- Verify SSL certificate validity
- Restart application if persistent

**Audio Processing Errors**
- Verify audio file format compatibility
- Check file size limits (recommended < 100MB)
- Ensure sufficient system memory
- Update audio codecs if needed

**Model Loading Issues**
- Verify sufficient disk space for models
- Check internet connection for initial download
- Ensure Python environment compatibility
- Clear model cache if corrupted

### Performance Tips

**For Large Audio Files**
- Use smaller Whisper models for faster processing
- Enable "Fast" performance mode
- Process files during low system usage
- Consider audio file compression

**For Real-time Usage**
- Use "tiny" or "base" Whisper models
- Enable audio caching
- Optimize system resources
- Close unnecessary applications

## Development and Customization

### Adding New Features

#### Custom Audio Processors
1. Implement processor interface in `src/core/audio_processor.py`
2. Add configuration options in Streamlit app
3. Update requirements and dependencies
4. Test with various audio formats

#### New AI Analysis Types
1. Extend `src/core/ai_insights_engine.py`
2. Add UI components in Streamlit app
3. Implement result visualization
4. Update configuration options

#### Additional Audio Sources
1. Create new ingester class in `src/data/`
2. Implement standard interface methods
3. Add configuration in main application
4. Test integration and error handling

### Code Structure

```
BLUEBLEAP_AI/
├── src/
│   ├── core/           # Core processing engines
│   ├── data/           # Data ingestion and management
│   ├── models/         # AI model wrappers
│   └── web/           # Web interface
├── tests/              # Unit and integration tests
├── docs/              # Documentation
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Testing

#### Unit Tests
```bash
python -m pytest tests/unit/
```

#### Integration Tests
```bash
python -m pytest tests/integration/
```

#### Performance Tests
```bash
python -m pytest tests/performance/
```

## System Requirements

### Minimum Requirements
- **OS**: Windows 10 or higher
- **Python**: 3.8+
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Network**: Stable internet connection

### Recommended Requirements
- **OS**: Windows 11
- **Python**: 3.9+
- **RAM**: 8GB or higher
- **Storage**: 5GB free space
- **Network**: High-speed internet connection
- **Audio**: High-quality microphone for recording

### Hardware Considerations
- **CPU**: Multi-core processor for faster processing
- **GPU**: CUDA-compatible GPU for accelerated Whisper processing
- **Audio**: Professional audio interface for high-quality recording
- **Storage**: SSD for faster file access and processing

## Security and Privacy

### Data Handling
- Audio files processed locally by default
- No audio data transmitted to external servers
- Temporary files automatically cleaned up
- User data privacy maintained

### Network Security
- SSL/TLS encryption for all network communications
- Secure connection handling and validation
- Automatic certificate verification
- Protected against common network attacks

### Access Control
- Local application access only
- No user authentication required
- Session-based security
- Automatic timeout and cleanup

## Future Enhancements

### Planned Features
- **Multi-language Support**: Additional language transcription
- **Real-time Streaming**: Live audio processing capabilities
- **Advanced Analytics**: Deep learning-based insights
- **Cloud Integration**: Optional cloud processing options
- **Mobile Support**: Responsive mobile interface

### Performance Improvements
- **GPU Acceleration**: CUDA support for faster processing
- **Model Optimization**: Quantized and optimized models
- **Parallel Processing**: Multi-threaded audio processing
- **Memory Optimization**: Efficient memory management

### User Experience
- **Customizable Interface**: User-defined layouts and themes
- **Batch Processing**: Multiple file processing capabilities
- **Export Options**: Various output format support
- **Integration APIs**: REST API for external applications

## Support and Contributing

### Getting Help
- Check this README for common solutions
- Review error logs in the application
- Test with different audio files and settings
- Verify system requirements and dependencies

### Contributing
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request
5. Follow coding standards and documentation

### Reporting Issues
- Provide detailed error descriptions
- Include system information and logs
- Specify audio file types and sizes
- Describe steps to reproduce

## License and Acknowledgments

### License
This project is licensed under the MIT License - see LICENSE file for details.

### Acknowledgments
- OpenAI for the Whisper speech recognition models
- LibriSpeech for the audio dataset
- Streamlit for the web application framework
- Open source community for various dependencies

### Third-party Libraries
- **Whisper**: OpenAI's speech recognition system
- **Streamlit**: Web application framework
- **aiohttp**: Asynchronous HTTP client/server
- **SoundFile**: Audio file processing
- **NumPy**: Numerical computing
- **Plotly**: Data visualization

## Version History

### Current Version: 1.0.0
- Initial release with core functionality
- LibriSpeech dataset integration
- Basic AI analysis capabilities
- Streamlit web interface

### Previous Versions
- **Beta 0.9.0**: Core transcription functionality
- **Alpha 0.5.0**: Basic audio processing
- **Prototype 0.1.0**: Initial concept and design

## Contact Information

For questions, issues, or contributions:
- **Repository**: [GitHub Repository URL]
- **Issues**: [GitHub Issues Page]
- **Documentation**: [Documentation URL]
- **Support**: [Support Email/Contact]

---

**Note**: This system is designed for research, development, and educational purposes. Ensure compliance with local regulations and data protection laws when processing audio content.




