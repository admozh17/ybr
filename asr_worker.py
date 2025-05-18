#!/usr/bin/env python3
"""ASR worker microservice for parallel speech transcription."""

import os
import tempfile
import pathlib
import json
import time
from typing import List, Dict, Any, Tuple, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import torch
import whisperx
from extractor import extract_audio, detect_speech_segments
from concurrent.futures import ProcessPoolExecutor

app = FastAPI(title="ASR Microservice")

# Configure maximum workers based on available CPU cores
import multiprocessing
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)

# Model cache
asr_models = {}

class TranscriptionRequest(BaseModel):
    """Request model for ASR transcription."""
    audio_url: str = Field(..., description="URL or local path to audio file")
    segments: List[Tuple[float, float]] = Field(
        default=None, 
        description="List of [start, end] segments to transcribe (in seconds)"
    )
    model_size: str = Field(
        default="small", 
        description="Whisper model size (tiny, small, medium, large, large-v2)"
    )

class TranscriptionResponse(BaseModel):
    """Response model for ASR transcription."""
    text: str = Field(..., description="Transcribed text")
    segments: List[Dict[str, Any]] = Field(..., description="Detailed segment information")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="Model used for transcription")

def load_asr_model(model_size: str = "small"):
    """Load and cache ASR model."""
    global asr_models
    
    if model_size in asr_models:
        return asr_models[model_size]
    
    print(f"🔄 Loading Whisper {model_size} model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    # For small model on CPU, use int8 for better performance
    if model_size == "small" and device == "cpu":
        compute_type = "int8"
    
    model = whisperx.load_model(model_size, device=device, compute_type=compute_type)
    asr_models[model_size] = model
    
    print(f"✅ Loaded Whisper {model_size} model on {device} using {compute_type}")
    return model

def process_segment(segment_path: str, model_size: str) -> Dict:
    """Process a single audio segment with Whisper."""
    try:
        model = load_asr_model(model_size)
        result = model.transcribe(segment_path)
        return result
    except Exception as e:
        print(f"❌ Error processing segment {segment_path}: {e}")
        return {"segments": [], "error": str(e)}

def transcribe_parallel(audio_path: pathlib.Path, segments: List[Tuple[float, float]], model_size: str):
    """Transcribe audio segments in parallel using ProcessPoolExecutor."""
    if not segments:
        # If no segments provided, transcribe the entire audio
        model = load_asr_model(model_size)
        return model.transcribe(str(audio_path))
    
    temp_dir = tempfile.mkdtemp()
    temp_path = pathlib.Path(temp_dir)
    segment_files = []
    
    # Extract each segment to a separate file
    for i, (start_time, end_time) in enumerate(segments):
        segment_path = temp_path / f"segment_{i}.wav"
        duration = end_time - start_time
        
        # Skip very short segments
        if duration < 0.3:
            continue
            
        try:
            # Use ffmpeg to extract segment
            import subprocess
            cmd = [
                "ffmpeg",
                "-i", str(audio_path),
                "-ss", str(start_time),
                "-t", str(duration),
                "-c:a", "pcm_s16le",
                str(segment_path),
                "-y"
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            segment_files.append((i, str(segment_path), start_time, end_time))
        except Exception as e:
            print(f"❌ Error extracting segment {i}: {e}")
    
    all_segments = []
    all_text = []
    
    # Process segments in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create tasks for parallel processing
        futures = []
        for i, segment_path, _, _ in segment_files:
            future = executor.submit(process_segment, segment_path, model_size)
            futures.append((i, future))
        
        # Process results as they complete
        for i, future in futures:
            try:
                result = future.result()
                if "error" in result:
                    print(f"⚠️ Segment {i} processing error: {result['error']}")
                    continue
                    
                # Get segment timestamps
                _, _, start_offset, end_offset = segment_files[i]
                
                # Adjust segment timestamps based on original offsets
                for seg in result["segments"]:
                    seg["start"] += start_offset
                    seg["end"] += start_offset
                    all_text.append(seg["text"])
                    
                all_segments.extend(result["segments"])
            except Exception as e:
                print(f"❌ Error processing future for segment {i}: {e}")
    
    # Sort segments by start time
    all_segments.sort(key=lambda x: x["start"])
    
    # Merge overlapping segments
    merged_segments = []
    if all_segments:
        current = all_segments[0]
        for next_seg in all_segments[1:]:
            # If segments overlap significantly
            if next_seg["start"] < current["end"] - 0.1:
                # Check which segment is longer/more confident
                if next_seg["end"] - next_seg["start"] > current["end"] - current["start"]:
                    # Keep the next segment as it's longer
                    current = next_seg
            else:
                merged_segments.append(current)
                current = next_seg
        merged_segments.append(current)
    
    # Clean up temporary files
    for _, path, _, _ in segment_files:
        try:
            os.remove(path)
        except:
            pass
    
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    # Construct final result
    return {
        "segments": merged_segments,
        "text": " ".join(all_text)
    }

@app.post("/asr", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest, background_tasks: BackgroundTasks):
    """Transcribe audio with Whisper."""
    start_time = time.time()
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            
            # Handle both local paths and remote URLs
            if request.audio_url.startswith(("http://", "https://")):
                import requests
                audio_file = temp_path / "audio.wav"
                with requests.get(request.audio_url, stream=True) as r:
                    r.raise_for_status()
                    with open(audio_file, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            else:
                # Assume it's a local path
                audio_file = pathlib.Path(request.audio_url)
                
            if not audio_file.exists():
                raise HTTPException(status_code=404, detail="Audio file not found")
            
            # Detect speech segments if not provided
            segments = request.segments
            if not segments:
                audio_path = extract_audio(audio_file)
                segments = detect_speech_segments(audio_path)
            
            # Process audio with parallel transcription
            result = transcribe_parallel(audio_file, segments, request.model_size)
            
            # Format response
            response = {
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
                "processing_time": time.time() - start_time,
                "model_used": request.model_size
            }
            
            return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)