"""
Main FastAPI application for STT system.

This module provides a RESTful API interface for speech-to-text functionality,
including direct web ingestion from LibriSpeech and local audio processing.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from ..core.stt_engine import STTEngine
from ..data.librispeech_ingester import LibriSpeechIngester
from ..utils.config import get_config
from ..utils.logging import setup_logger

# Setup logging
logger = setup_logger(__name__)
config = get_config()

# Global instances
stt_engine: Optional[STTEngine] = None
librispeech_ingester: Optional[LibriSpeechIngester] = None

# Global variable to store batch processing status
batch_processing_status = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global stt_engine, librispeech_ingester
    
    # Startup
    logger.info("Starting STT system...")
    
    try:
        # Initialize STT engine
        stt_engine = STTEngine(config)
        logger.info("STT Engine initialized successfully")
        
        # Initialize LibriSpeech ingester with STT engine
        librispeech_ingester = LibriSpeechIngester(config, stt_engine)
        await librispeech_ingester.__aenter__()
        logger.info("LibriSpeech Ingester initialized successfully")
        
        logger.info("STT system startup completed")
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down STT system...")
        
        try:
            if stt_engine:
                await stt_engine.cleanup()
            
            if librispeech_ingester:
                await librispeech_ingester.cleanup()
                
            logger.info("STT system shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")


# Create FastAPI application
app = FastAPI(
    title="Lightweight STT System with Web Data Ingestion",
    description="Fast Speech-to-Text system with minimal dependencies - processes audio directly from web sources",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web interface
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - serve the main web interface."""
    try:
        with open("src/web/static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>Speech-to-Insights AI System</h1>
                <p>Web interface not found. Please check the static files.</p>
                <p><a href="/docs">API Documentation</a></p>
            </body>
        </html>
        """)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        if stt_engine is None or librispeech_ingester is None:
            raise HTTPException(status_code=503, detail="System not ready")
        
        # Check system components
        model_info = stt_engine.get_model_info()
        stats = await librispeech_ingester.get_processing_stats()
        
        return {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "model_info": model_info,
            "processing_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.get("/info")
async def system_info():
    """Get system information and configuration."""
    try:
        return {
            "system": "STT System with Web Data Ingestion",
            "version": "1.0.0",
            "configuration": config.to_dict(),
            "endpoints": {
                "transcribe": "POST /transcribe",
                "transcribe_url": "POST /transcribe_url",
                "librispeech_subsets": "GET /librispeech/subsets",
                "librispeech_metadata": "GET /librispeech/metadata/{subset}",
                "librispeech_transcribe": "POST /librispeech/transcribe",
                "health": "GET /health",
                "info": "GET /info"
            }
        }
        
    except Exception as e:
        logger.error(f"System info failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System info failed: {str(e)}")


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    apply_noise_reduction: bool = Form(True),
    enhance_audio: bool = Form(True),
    post_process: bool = Form(True)
):
    """
    Transcribe uploaded audio file.
    
    Args:
        file: Audio file to transcribe
        language: Expected language code (auto-detected if None)
        apply_noise_reduction: Whether to apply noise reduction
        enhance_audio: Whether to enhance audio quality
        post_process: Whether to apply post-processing
    """
    try:
        if stt_engine is None:
            raise HTTPException(status_code=503, detail="STT Engine not ready")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read file content
        audio_data = await file.read()
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"Processing audio file: {file.filename}, size: {len(audio_data)} bytes")
        
        # Transcribe audio
        result = await stt_engine.transcribe_audio(
            audio_data=audio_data,
            language=language,
            apply_noise_reduction=apply_noise_reduction,
            enhance_audio=enhance_audio,
            post_process=post_process
        )
        
        # Convert result to dict for JSON response
        response_data = {
            "text": result.text,
            "confidence": result.confidence,
            "language": result.language,
            "duration": result.duration,
            "processing_time": result.processing_time,
            "audio_quality_score": result.audio_quality_score,
            "noise_reduction_applied": result.noise_reduction_applied,
            "word_timestamps": result.word_timestamps,
            "source_file": file.filename,
            "file_size": len(audio_data)
        }
        
        return JSONResponse(content=response_data, status_code=200)
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/transcribe_url")
async def transcribe_from_url(
    url: str = Form(...),
    language: Optional[str] = Form(None),
    apply_noise_reduction: bool = Form(True)
):
    """
    Transcribe audio directly from URL.
    
    Args:
        url: URL of the audio file
        language: Expected language code (auto-detected if None)
        apply_noise_reduction: Whether to apply noise reduction
    """
    try:
        if librispeech_ingester is None:
            raise HTTPException(status_code=503, detail="LibriSpeech Ingester not ready")
        
        logger.info(f"Transcribing audio from URL: {url}")
        
        # Transcribe from URL
        result = await librispeech_ingester.transcribe_from_url(
            audio_url=url,
            language=language,
            apply_noise_reduction=apply_noise_reduction
        )
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        logger.error(f"URL transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"URL transcription failed: {str(e)}")


@app.get("/librispeech/subsets")
async def get_librispeech_subsets():
    """Get available LibriSpeech subsets."""
    try:
        if librispeech_ingester is None:
            raise HTTPException(status_code=503, detail="LibriSpeech Ingester not ready")
        
        subsets = await librispeech_ingester.get_available_subsets()
        
        return {
            "subsets": subsets,
            "total_count": len(subsets),
            "description": "Available LibriSpeech dataset subsets"
        }
        
    except Exception as e:
        logger.error(f"Failed to get subsets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get subsets: {str(e)}")


@app.get("/librispeech/metadata/{subset}")
async def get_librispeech_metadata(subset: str):
    """Get metadata for a specific LibriSpeech subset."""
    try:
        if librispeech_ingester is None:
            raise HTTPException(status_code=503, detail="LibriSpeech Ingester not ready")
        
        metadata = await librispeech_ingester.get_subset_metadata(subset)
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to get metadata for {subset}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")


@app.post("/librispeech/transcribe")
async def transcribe_librispeech_subset(
    subset: str = Form(...),
    max_files: int = Form(5),
    language: Optional[str] = Form(None)
):
    """
    Batch transcribe files from a LibriSpeech subset.
    
    Args:
        subset: Subset name (e.g., 'dev-clean', 'test-clean')
        max_files: Maximum number of files to process
        language: Expected language code
    """
    try:
        if librispeech_ingester is None:
            raise HTTPException(status_code=503, detail="LibriSpeech Ingester not ready")
        
        logger.info(f"Received batch transcription request: subset={subset}, max_files={max_files}, language={language}")
        
        # Generate unique job ID
        job_id = f"batch_{subset}_{int(asyncio.get_event_loop().time())}"
        
        # Initialize job status
        batch_processing_status[job_id] = {
            "job_id": job_id,
            "subset": subset,
            "max_files": max_files,
            "language": language,
            "status": "starting",
            "start_time": asyncio.get_event_loop().time(),
            "total_files": 0,
            "processed_files": 0,
            "results": [],
            "error": None
        }
        
        logger.info(f"Job {job_id} initialized and added to batch_processing_status")
        
        # Start background task for batch processing - RUN DIRECTLY INSTEAD OF BackgroundTasks
        logger.info(f"Starting background task directly for job {job_id}")
        
        # Create a task and run it in the background
        task = asyncio.create_task(
            process_batch_transcription(
                job_id=job_id,
                subset=subset,
                max_files=max_files,
                language=language
            )
        )
        
        # Store the task reference for potential cancellation
        batch_processing_status[job_id]["task"] = task
        
        # Add error handling for the task
        def task_done_callback(task_future):
            try:
                # Get the result (this will raise any exception that occurred)
                task_future.result()
                logger.info(f"Background task for job {job_id} completed successfully")
            except Exception as e:
                logger.error(f"Background task for job {job_id} failed: {str(e)}")
                # Update job status to failed
                if job_id in batch_processing_status:
                    batch_processing_status[job_id].update({
                        "status": "failed",
                        "error": str(e),
                        "end_time": asyncio.get_event_loop().time()
                    })
        
        # Add callback to handle task completion
        task.add_done_callback(task_done_callback)
        
        logger.info(f"Background task created and started for job {job_id}")
        
        return {
            "message": f"Batch transcription started for subset: {subset}",
            "job_id": job_id,
            "subset": subset,
            "max_files": max_files,
            "status": "processing",
            "note": "Use /batch/status/{job_id} to track progress",
            "tracking_url": f"/batch/status/{job_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start batch transcription: {str(e)}")
        logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch transcription failed: {str(e)}")


@app.get("/librispeech/search")
async def search_librispeech_files(
    subset: str,
    query: str,
    max_results: int = 20
):
    """
    Search for audio files in a LibriSpeech subset.
    
    Args:
        subset: Subset name
        query: Search query
        max_results: Maximum number of results
    """
    try:
        if librispeech_ingester is None:
            raise HTTPException(status_code=503, detail="LibriSpeech Ingester not ready")
        
        results = await librispeech_ingester.search_audio_files(
            subset=subset,
            query=query,
            max_results=max_results
        )
        
        return {
            "query": query,
            "subset": subset,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/librispeech/real-sample")
async def get_real_librispeech_sample(
    subset: str = Form("dev-clean"),
    file_index: int = Form(0)
):
    """
    Get a real LibriSpeech sample with actual transcription.
    
    Args:
        subset: Subset name (default: dev-clean)
        file_index: Index of the file to extract (default: 0)
    """
    try:
        if librispeech_ingester is None:
            raise HTTPException(status_code=503, detail="LibriSpeech Ingester not ready")
        
        logger.info(f"Requesting real LibriSpeech sample from {subset}, file_index={file_index}")
        
        # Get real LibriSpeech sample with specific file index
        result = await librispeech_ingester.get_real_librispeech_sample(subset, file_index)
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        logger.error(f"Real LibriSpeech sample failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Real sample failed: {str(e)}")


@app.get("/stats")
async def get_processing_stats():
    """Get system processing statistics."""
    try:
        if librispeech_ingester is None:
            raise HTTPException(status_code=503, detail="LibriSpeech Ingester not ready")
        
        stats = await librispeech_ingester.get_processing_stats()
        
        return {
            "system_stats": stats,
            "timestamp": asyncio.get_event_loop().time(),
            "configuration": config.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# New batch tracking endpoints
@app.get("/batch/status")
async def get_batch_status():
    """Get current batch processing status."""
    try:
        return {
            "batch_status": batch_processing_status,
            "active_jobs": len(batch_processing_status),
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"Failed to get batch status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get batch status: {str(e)}")


@app.get("/batch/status/{job_id}")
async def get_batch_job_status(job_id: str):
    """Get status of a specific batch job."""
    try:
        if job_id not in batch_processing_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return batch_processing_status[job_id]
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@app.get("/batch/results/{job_id}")
async def get_batch_results(job_id: str):
    """Get results of a completed batch job."""
    try:
        if job_id not in batch_processing_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_info = batch_processing_status[job_id]
        if job_info["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job not completed yet")
        
        return {
            "job_id": job_id,
            "results": job_info.get("results", []),
            "summary": {
                "total_files": job_info.get("total_files", 0),
                "processed_files": len(job_info.get("results", [])),
                "processing_time": job_info.get("processing_time", 0)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get batch results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get batch results: {str(e)}")


# Add debug endpoint for batch processing
@app.get("/batch/debug")
async def debug_batch_processing():
    """Debug endpoint to check batch processing status."""
    try:
        return {
            "batch_processing_status": batch_processing_status,
            "active_jobs": len(batch_processing_status),
            "job_details": {
                job_id: {
                    "status": job_info["status"],
                    "subset": job_info["subset"],
                    "max_files": job_info["max_files"],
                    "total_files": job_info["total_files"],
                    "processed_files": job_info["processed_files"],
                    "start_time": job_info["start_time"],
                    "error": job_info.get("error")
                }
                for job_id, job_info in batch_processing_status.items()
            },
            "timestamp": asyncio.get_event_loop().time(),
            "librispeech_ingester_ready": librispeech_ingester is not None
        }
    except Exception as e:
        logger.error(f"Debug endpoint failed: {str(e)}")
        return {
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time()
        }


# Add endpoint to cancel batch tasks
@app.post("/batch/cancel/{job_id}")
async def cancel_batch_task(job_id: str):
    """Cancel a running batch task."""
    try:
        if job_id not in batch_processing_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_info = batch_processing_status[job_id]
        
        if job_info["status"] in ["completed", "failed"]:
            raise HTTPException(status_code=400, detail="Cannot cancel completed or failed job")
        
        # Cancel the task if it exists
        if "task" in job_info and not job_info["task"].done():
            job_info["task"].cancel()
            logger.info(f"Task for job {job_id} cancelled")
        
        # Update job status
        job_info.update({
            "status": "cancelled",
            "end_time": asyncio.get_event_loop().time()
        })
        
        return {
            "message": f"Job {job_id} cancelled successfully",
            "job_id": job_id,
            "status": "cancelled"
        }
        
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


async def process_batch_transcription(job_id: str, subset: str, max_files: int, language: Optional[str] = None):
    """Background task for processing batch transcription."""
    try:
        logger.info(f"Starting background batch processing for job {job_id}")
        
        # Update status to processing
        batch_processing_status[job_id]["status"] = "processing"
        logger.info(f"Job {job_id} status updated to processing")
        
        # Get subset metadata
        logger.info(f"Getting metadata for subset: {subset}")
        metadata = await librispeech_ingester.get_subset_metadata(subset)
        logger.info(f"Metadata retrieved: {metadata}")
        
        # Extract total files from metadata
        total_files = metadata.get("total_files", 0)
        if total_files == 0:
            # If metadata doesn't have total_files, use max_files as fallback
            total_files = max_files
            logger.warning(f"total_files not found in metadata, using max_files: {total_files}")
        
        # Limit to max_files
        total_files = min(max_files, total_files)
        logger.info(f"Processing {total_files} files for job {job_id}")
        
        # Update total files in job status
        batch_processing_status[job_id]["total_files"] = total_files
        
        # Process files (simulate for now)
        results = []
        for i in range(total_files):
            logger.info(f"Processing file {i+1}/{total_files} for job {job_id}")
            
            # Update progress
            batch_processing_status[job_id]["processed_files"] = i + 1
            
            # Simulate processing time
            await asyncio.sleep(1)  # Simulate 1 second per file
            
            # Add mock result
            results.append({
                "file_index": i,
                "file_name": f"file_{i}.flac",
                "transcription": f"Sample transcription for file {i} from {subset}",
                "confidence": 0.85,
                "processing_time": 1.0
            })
            
            logger.info(f"File {i+1}/{total_files} processed for job {job_id}")
        
        # Calculate processing time
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - batch_processing_status[job_id]["start_time"]
        
        # Update final status
        batch_processing_status[job_id].update({
            "status": "completed",
            "results": results,
            "processing_time": processing_time,
            "end_time": end_time
        })
        
        logger.info(f"Batch transcription completed for job {job_id}: {len(results)} files processed in {processing_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Batch transcription failed for job {job_id}: {str(e)}")
        logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
        
        # Update error status
        try:
            batch_processing_status[job_id].update({
                "status": "failed",
                "error": str(e),
                "end_time": asyncio.get_event_loop().time()
            })
            logger.info(f"Job {job_id} status updated to failed")
        except Exception as update_error:
            logger.error(f"Failed to update job {job_id} error status: {str(update_error)}")
        
        # Re-raise the exception for debugging
        raise


# Add new endpoints for real audio processing
@app.get("/librispeech/real-audio/{subset}")
async def get_real_audio_info(subset: str):
    """Get real audio file information for a LibriSpeech subset."""
    try:
        if librispeech_ingester is None:
            raise HTTPException(status_code=503, detail="LibriSpeech Ingester not ready")
        
        audio_info = await librispeech_ingester.get_real_audio_info(subset)
        return audio_info
        
    except Exception as e:
        logger.error(f"Failed to get real audio info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get real audio info: {str(e)}")

@app.post("/librispeech/transcribe-real")
async def transcribe_real_audio(
    subset: str = Form(...),
    file_index: int = Form(0),
    language: Optional[str] = Form(None)
):
    """
    Transcribe real LibriSpeech audio file with streaming.
    
    Args:
        subset: Subset name (e.g., 'dev-clean')
        file_index: Index of the file to transcribe
        language: Expected language code
    """
    try:
        if librispeech_ingester is None:
            raise HTTPException(status_code=503, detail="LibriSpeech Ingester not ready")
        
        logger.info(f"Received real audio transcription request: subset={subset}, file_index={file_index}, language={language}")
        
        # Transcribe real audio
        result = await librispeech_ingester.transcribe_real_audio(
            subset=subset,
            file_index=file_index,
            language=language
        )
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        logger.error(f"Real audio transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Real audio transcription failed: {str(e)}")

@app.get("/librispeech/stream-audio/{subset}/{file_index}")
async def stream_real_audio(subset: str, file_index: int):
    """
    Stream real LibriSpeech audio file.
    
    Args:
        subset: Subset name
        file_index: Index of the file to stream
    """
    try:
        if librispeech_ingester is None:
            raise HTTPException(status_code=503, detail="LibriSpeech Ingester not ready")
        
        logger.info(f"Starting real audio stream: subset={subset}, file_index={file_index}")
        
        async def audio_generator():
            async for chunk in librispeech_ingester.stream_audio_file(subset, file_index):
                yield chunk
        
        return StreamingResponse(
            audio_generator(),
            media_type="audio/flac",
            headers={
                "Content-Disposition": f"attachment; filename={subset}_file_{file_index}.flac"
            }
        )
        
    except Exception as e:
        logger.error(f"Real audio streaming failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Real audio streaming failed: {str(e)}")


# Add new endpoints for speech-to-insights functionality
@app.post("/transcribe-with-insights")
async def transcribe_with_insights(
    file: UploadFile = File(...),
    analysis_type: str = Form("summary"),
    language: Optional[str] = Form(None),
    apply_noise_reduction: bool = Form(True),
    enhance_audio: bool = Form(True),
    post_process: bool = Form(True),
    custom_prompt: Optional[str] = Form(None),
    context: Optional[str] = Form(None)
):
    """
    Transcribe audio and generate AI-powered insights in one operation.
    
    Args:
        file: Audio file to transcribe
        analysis_type: Type of AI analysis (summary, qa, insights, sentiment, actions, topics)
        language: Expected language code (auto-detected if None)
        apply_noise_reduction: Whether to apply noise reduction
        enhance_audio: Whether to enhance audio quality
        post_process: Whether to apply post-processing
        custom_prompt: Custom prompt for AI analysis
        context: Additional context for AI analysis
    """
    try:
        if stt_engine is None:
            raise HTTPException(status_code=503, detail="STT Engine not ready")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read file content
        audio_data = await file.read()
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"Processing audio file with insights: {file.filename}, analysis_type: {analysis_type}")
        
        # Transcribe with insights
        result = await stt_engine.transcribe_with_insights(
            audio_data=audio_data,
            analysis_type=analysis_type,
            language=language,
            apply_noise_reduction=apply_noise_reduction,
            enhance_audio=enhance_audio,
            post_process=post_process,
            custom_prompt=custom_prompt,
            context=context
        )
        
        # Add file metadata
        result["source_file"] = file.filename
        result["file_size"] = len(audio_data)
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        logger.error(f"Speech-to-insights failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech-to-insights failed: {str(e)}")

@app.post("/librispeech/transcribe-with-insights")
async def transcribe_librispeech_with_insights(
    subset: str = Form(...),
    file_index: int = Form(0),
    analysis_type: str = Form("summary"),
    language: Optional[str] = Form(None),
    custom_prompt: Optional[str] = Form(None),
    context: Optional[str] = Form(None)
):
    """
    Transcribe LibriSpeech audio and generate AI insights.
    
    Args:
        subset: Subset name (e.g., 'dev-clean')
        file_index: Index of the file to transcribe
        analysis_type: Type of AI analysis
        language: Expected language code
        custom_prompt: Custom prompt for analysis
        context: Additional context for analysis
    """
    try:
        if librispeech_ingester is None:
            raise HTTPException(status_code=503, detail="LibriSpeech Ingester not ready")
        
        logger.info(f"Starting LibriSpeech speech-to-insights: subset={subset}, file_index={file_index}, analysis_type={analysis_type}")
        
        # Transcribe real audio with insights
        result = await librispeech_ingester.transcribe_real_audio_with_insights(
            subset=subset,
            file_index=file_index,
            analysis_type=analysis_type,
            language=language,
            custom_prompt=custom_prompt,
            context=context
        )
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        logger.error(f"LibriSpeech speech-to-insights failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LibriSpeech speech-to-insights failed: {str(e)}")

@app.post("/batch/transcribe-with-insights")
async def batch_transcribe_with_insights(
    files: List[UploadFile] = File(...),
    analysis_type: str = Form("summary"),
    language: Optional[str] = Form(None),
    apply_noise_reduction: bool = Form(True),
    enhance_audio: bool = Form(True),
    post_process: bool = Form(True)
):
    """
    Batch transcribe multiple audio files with AI insights generation.
    
    Args:
        files: List of audio files to process
        analysis_type: Type of AI analysis to perform
        language: Expected language code
        apply_noise_reduction: Whether to apply noise reduction
        enhance_audio: Whether to enhance audio quality
        post_process: Whether to apply post-processing
    """
    try:
        if stt_engine is None:
            raise HTTPException(status_code=503, detail="STT Engine not ready")
        
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        logger.info(f"Starting batch speech-to-insights for {len(files)} files")
        
        # Read all files
        audio_data_list = []
        for file in files:
            if not file.content_type or not file.content_type.startswith("audio/"):
                continue
            
            audio_data = await file.read()
            if audio_data:
                audio_data_list.append(audio_data)
        
        if not audio_data_list:
            raise HTTPException(status_code=400, detail="No valid audio files found")
        
        # Process with insights
        result = await stt_engine.batch_transcribe_with_insights(
            audio_files=audio_data_list,
            analysis_type=analysis_type,
            language=language,
            apply_noise_reduction=apply_noise_reduction,
            enhance_audio=enhance_audio,
            post_process=post_process
        )
        
        return JSONResponse(content=result, status_code=200)
        
    except Exception as e:
        logger.error(f"Batch speech-to-insights failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch speech-to-insights failed: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        "src.web.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level=config.log_level.lower()
    )


if __name__ == "__main__":
    run_server()

@app.post("/summarize")
async def summarize_text(
    text: str = Form(...),
    max_length: int = Form(100),
    style: str = Form("concise")  # concise, detailed, bullet_points
):
    """
    Summarize text using OpenAI API.
    
    Args:
        text: Text to summarize
        max_length: Maximum length of summary
        style: Summary style (concise, detailed, bullet_points)
    """
    try:
        import openai
        import os
        
        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        openai.api_key = api_key
        
        logger.info(f"Summarizing text: {len(text)} characters, max_length={max_length}, style={style}")
        
        # Create prompt based on style
        if style == "concise":
            prompt = f"Summarize the following text in {max_length} words or less:\n\n{text}"
        elif style == "detailed":
            prompt = f"Provide a detailed summary of the following text in {max_length} words or less:\n\n{text}"
        elif style == "bullet_points":
            prompt = f"Summarize the following text in {max_length} words or less using bullet points:\n\n{text}"
        else:
            prompt = f"Summarize the following text in {max_length} words or less:\n\n{text}"
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise, accurate summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_length * 2,  # Allow some flexibility
            temperature=0.3  # Low temperature for consistent summaries
        )
        
        summary = response.choices[0].message.content.strip()
        
        # Calculate compression ratio
        original_length = len(text.split())
        summary_length = len(summary.split())
        compression_ratio = summary_length / original_length if original_length > 0 else 0
        
        response_data = {
            "original_text": text,
            "summary": summary,
            "original_length": original_length,
            "summary_length": summary_length,
            "compression_ratio": round(compression_ratio, 3),
            "max_length": max_length,
            "style": style,
            "model": "gpt-3.5-turbo",
            "api_provider": "openai"
        }
        
        logger.info(f"Summarization completed: {original_length} → {summary_length} words")
        return JSONResponse(content=response_data, status_code=200)
        
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
