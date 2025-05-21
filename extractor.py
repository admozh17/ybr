import subprocess, os, pathlib, cv2, pytesseract, torch, googlemaps
import whisperx
from tqdm import tqdm
from yt_dlp import YoutubeDL
import subprocess
import pathlib
import whisperx
import torch
import re
import numpy as np
from pydub import AudioSegment
import time

# Import parallel_ocr instead of text_region_detector
from parallel_ocr import parallel_ocr as optimized_ocr

# ---------- Download ----------
def fetch_clip(url: str, out_path: pathlib.Path):
    """Download video via yt‑dlp (supports IG, TikTok, YT)."""
    cmd = ["yt-dlp", "-f", "mp4", "-o", str(out_path), url]
    subprocess.run(cmd, check=True)


# ---------- Speech ----------
import subprocess
import pathlib
import whisperx
import torch
import re
import numpy as np
from pydub import AudioSegment
import time


def extract_audio(video_path: pathlib.Path) -> pathlib.Path:
    """Extract audio from video file."""
    # Add debug print
    print(f"🔍 extract_audio called with: {video_path} (suffix: {video_path.suffix})")
    
    # Check if this is already an audio file
    if video_path.suffix.lower() == ".wav":
        print(f"⚠️ Input is already a WAV file, returning as is.")
        return video_path
        
    # Continue with normal extraction
    audio_path = video_path.with_suffix(".wav")
    
    # Add safety check - don't try to extract if input and output are the same file
    if str(video_path) == str(audio_path):
        print(f"⚠️ Input and output paths are identical: {video_path}")
        return audio_path
    
    # Add debug print
    print(f"🔊 Extracting audio to: {audio_path}")
    
    # Run the extraction
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                "-q:a",
                "0",
                "-map",
                "a",
                str(audio_path),
                "-y",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return audio_path
    except subprocess.CalledProcessError as e:
        print(f"❌ FFMPEG error extracting audio: {e}")
        # If extraction fails but the output file exists somehow, return it
        if audio_path.exists():
            print(f"⚠️ Using existing audio file despite extraction error")
            return audio_path
        raise  # Re-raise the error if we can't recover

def sample_audio(audio_path: pathlib.Path, duration_sec: int = 10) -> pathlib.Path:
    """Extract a sample from the beginning of the audio file."""
    sample_path = audio_path.with_name(f"{audio_path.stem}_sample{audio_path.suffix}")
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(audio_path),
            "-t",
            str(duration_sec),
            str(sample_path),
            "-y",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return sample_path


def detect_complexity(transcript: str) -> float:
    """
    Analyze transcript to determine audio complexity score.
    Returns a score between 0 (simple) and 1 (complex).
    """
    # Detect potential issues in transcript
    score = 0.0

    # Check for non-English content
    english_ratio = len(re.findall(r"[a-zA-Z]", transcript)) / max(len(transcript), 1)
    if english_ratio < 0.7:
        score += 0.5  # Likely non-English content

    # Check for gibberish or low confidence
    if "[" in transcript or "]" in transcript:
        score += 0.2  # Contains uncertainty markers

    # Check for repetition, a sign the model is struggling
    words = transcript.lower().split()
    if len(words) > 5:
        unique_words = len(set(words))
        repetition_ratio = unique_words / len(words)
        if repetition_ratio < 0.5:
            score += 0.3  # High repetition

    # Check for very short result (model couldn't detect much)
    if len(words) < 5 and len(transcript) > 10:
        score += 0.3

    return min(score, 1.0)


def detect_audio_quality(audio_path: pathlib.Path) -> float:
    """
    Analyze audio file to determine quality score.
    Returns a score between 0 (high quality) and 1 (low quality).
    """
    try:
        # Load audio
        audio = AudioSegment.from_file(audio_path)

        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())

        # Normalize
        samples = samples / np.max(np.abs(samples))

        # Calculate signal-to-noise ratio (simple approximation)
        noise_floor = np.sort(np.abs(samples))[int(len(samples) * 0.1)]
        snr = np.mean(np.abs(samples)) / (noise_floor + 1e-10)

        # Calculate dynamic range
        dynamic_range = np.max(samples) - np.min(samples)

        # Combine metrics into score (lower is better audio)
        score = 0.0

        if snr < 5:
            score += 0.4  # Low SNR

        if dynamic_range < 0.5:
            score += 0.3  # Low dynamic range

        # Check volume level
        if np.mean(np.abs(samples)) < 0.1:
            score += 0.3  # Very quiet audio

        return min(score, 1.0)
    except Exception as e:
        print(f"Error analyzing audio quality: {e}")
        return 0.5  # Default to moderate quality on error


def detect_speech_segments(
    audio_path: pathlib.Path, threshold: float = 0.03, min_duration: float = 1.0
):
    """Detect speech segments based on audio energy."""
    try:
        # Load audio
        audio = AudioSegment.from_file(audio_path)

        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Parameters
        chunk_size_ms = 500  # 500ms chunks
        segments = []
        current_segment = None

        # Process in chunks
        for i in range(0, len(audio), chunk_size_ms):
            chunk = audio[i : i + chunk_size_ms]

            # Get chunk RMS (root mean square - rough measure of loudness)
            rms = chunk.rms
            normalized_rms = rms / 32768.0  # Normalize to 0-1 range

            # If above threshold, consider it speech
            if normalized_rms > threshold:
                chunk_start_time = i / 1000.0  # Convert ms to seconds

                if current_segment is None:
                    # Start new segment
                    current_segment = [
                        chunk_start_time,
                        chunk_start_time + (chunk_size_ms / 1000.0),
                    ]
                else:
                    # Extend current segment
                    current_segment[1] = chunk_start_time + (chunk_size_ms / 1000.0)
            else:
                # If below threshold and we have a segment, add it
                if current_segment is not None:
                    # Only add if long enough
                    if current_segment[1] - current_segment[0] >= min_duration:
                        segments.append(tuple(current_segment))
                    current_segment = None

        # Add final segment if it exists
        if (
            current_segment is not None
            and current_segment[1] - current_segment[0] >= min_duration
        ):
            segments.append(tuple(current_segment))

        print(f"Found {len(segments)} potential speech segments")
        return segments
    except Exception as e:
        print(f"Error detecting speech segments: {e}")
        return []


def whisper_transcribe_segments(video_path: pathlib.Path) -> str:
    """Transcribe only speech segments with adaptive model selection."""
    print("🎤 Starting segmented transcription...")
    start_time = time.time()

    # Check if input is already an audio file
    is_audio = video_path.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    
    # Extract audio only if it's not already an audio file
    if is_audio:
        print(f"Input is already an audio file: {video_path}")
        audio_path = video_path
    else:
        # Extract audio from video
        audio_path = extract_audio(video_path)

    # Check duration first
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_sec = len(audio) / 1000.0
    except Exception as e:
        print(f"❌ Error reading audio file: {e}")
        print("Falling back to regular transcription")
        return adaptive_whisper_transcribe(video_path)

    # For short videos, just use regular adaptive transcription
    if duration_sec < 30:
        print(f"Short audio detected ({duration_sec:.1f}s), processing entire audio")
        return adaptive_whisper_transcribe(video_path)

    # Detect speech segments using energy-based approach
    print("🔍 Detecting speech segments...")
    speech_segments = detect_speech_segments(audio_path)

    # If no segments detected or very few, just process the whole thing
    if len(speech_segments) <= 2:
        print("Few speech segments detected, processing entire audio...")
        return adaptive_whisper_transcribe(video_path)

    # Calculate total speech duration
    total_speech_duration = sum(end - start for start, end in speech_segments)

    # If most of the video is speech anyway, just process it all
    if total_speech_duration > 0.8 * duration_sec:
        print(
            f"Audio is mostly speech ({total_speech_duration:.1f}s of {duration_sec:.1f}s), processing entire audio"
        )
        return adaptive_whisper_transcribe(video_path)

    print(
        f"Detected {len(speech_segments)} speech segments totaling {total_speech_duration:.1f}s (of {duration_sec:.1f}s)"
    )

    # Rest of the function remains the same...
    # Process a sample segment to determine complexity
    sample_segment = speech_segments[0]
    sample_path = audio_path.with_name(f"{audio_path.stem}_sample{audio_path.suffix}")

    # Extract sample segment
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(audio_path),
            "-ss",
            str(sample_segment[0]),
            "-to",
            str(sample_segment[1]),
            str(sample_path),
            "-y",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Continue with the rest of the function...
    # [rest of function code]
    
    # Analyze sample for complexity
    device = "cuda" if torch.cuda.is_available() else "cpu"
    small_model = whisperx.load_model("small", device=device, compute_type="int8")

    # Transcribe sample
    sample_result = small_model.transcribe(str(sample_path))
    sample_transcript = " ".join(seg["text"] for seg in sample_result["segments"])

    # Check complexity
    complexity_score = detect_complexity(sample_transcript)
    quality_score = detect_audio_quality(sample_path)
    use_large_model = complexity_score > 0.4 or quality_score > 0.5

    # Select model based on complexity
    if use_large_model:
        print("⚠️ Complex content detected, using large model for segments")
        model = whisperx.load_model("large-v2", device=device, compute_type="float16")
    else:
        print("✅ Standard content detected, using small model for segments")
        model = small_model

    # Process each segment
    all_transcripts = []
    for i, (start_time, end_time) in enumerate(speech_segments):
        print(
            f"Processing segment {i + 1}/{len(speech_segments)} ({start_time:.1f}s - {end_time:.1f}s)"
        )

        # Skip very short segments
        if end_time - start_time < 0.5:
            continue

        # Extract segment
        segment_path = audio_path.with_name(
            f"{audio_path.stem}_segment{i}{audio_path.suffix}"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(audio_path),
                "-ss",
                str(start_time),
                "-to",
                str(end_time),
                str(segment_path),
                "-y",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Transcribe segment
        try:
            result = model.transcribe(str(segment_path))
            segment_transcript = " ".join(seg["text"] for seg in result["segments"])

            # Only add non-empty segments
            if segment_transcript.strip():
                all_transcripts.append(segment_transcript)
        except Exception as e:
            print(f"Error processing segment {i}: {e}")

    # If we didn't get any transcripts, fall back to full processing
    if not all_transcripts:
        print("No valid transcripts from segments, falling back to full processing")
        return adaptive_whisper_transcribe(video_path)

    # Combine all segments
    full_transcript = " ".join(all_transcripts)

    end_time = time.time()
    print(
        f"✅ Segmented transcription completed in {end_time - start_time:.2f} seconds"
    )
    print(f"   Used large model: {use_large_model}")
    print(
        f"   Processed {len(speech_segments)} segments instead of entire {duration_sec:.1f}s audio"
    )

    return full_transcript


def adaptive_whisper_transcribe(
    video_path: pathlib.Path,
    complexity_threshold: float = 0.4,
    quality_threshold: float = 0.5,
) -> str:
    """
    Adaptively choose and apply the appropriate Whisper model based on content complexity.

    Args:
        video_path: Path to video or audio file
        complexity_threshold: Threshold above which to use large model
        quality_threshold: Audio quality threshold to use large model

    Returns:
        Transcribed text
    """
    print("🎤 Starting adaptive transcription...")
    start_time = time.time()

    # Check if input is already an audio file
    is_audio = video_path.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    
    # Extract audio only if it's not already an audio file
    if is_audio:
        print(f"Input is already an audio file: {video_path}")
        audio_path = video_path
    else:
        # Extract audio from video
        audio_path = extract_audio(video_path)

    try:
        # Take a short sample from the beginning for analysis
        sample_path = sample_audio(audio_path)

        # Check audio quality
        print("🔊 Analyzing audio quality...")
        quality_score = detect_audio_quality(sample_path)
        print(
            f"   Audio quality score: {quality_score:.2f} (0=high quality, 1=low quality)"
        )

        # Start with small model on the sample
        print("🔍 Running initial transcription with small model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        small_model = whisperx.load_model("small", device=device, compute_type="int8")

        # Transcribe sample with small model
        sample_result = small_model.transcribe(str(sample_path))
        sample_transcript = " ".join(seg["text"] for seg in sample_result["segments"])

        # Analyze transcript complexity
        complexity_score = detect_complexity(sample_transcript)
        print(f"   Content complexity score: {complexity_score:.2f} (0=simple, 1=complex)")

        # Decide which model to use for full transcription
        use_large_model = (
            complexity_score > complexity_threshold or quality_score > quality_threshold
        )

        if use_large_model:
            print("⚠️ Complex content or low audio quality detected!")
            print("🔄 Switching to large-v2 model for better accuracy...")

            model = whisperx.load_model(
                "large-v2",
                device=device,
                compute_type="float16",  # Slight compromise between speed and accuracy
            )
        else:
            print("✅ Content appears to be standard speech. Using small model...")
            model = small_model  # Reuse already loaded small model

        # Transcribe the full audio
        print("📝 Transcribing full audio...")
        result = model.transcribe(str(audio_path))
        transcript = " ".join(seg["text"] for seg in result["segments"])

        # Rest of function remains the same...
        # [rest of function code]
        
        end_time = time.time()
        print(f"✅ Transcription completed in {end_time - start_time:.2f} seconds")
        print(f"   Used large model: {use_large_model}")

        return transcript
        
    except Exception as e:
        print(f"❌ Error in adaptive transcription: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a blank transcript on error rather than crashing
        return ""


# For backward compatibility, keep the original function name
def whisper_transcribe(video_path: pathlib.Path) -> str:
    """Original function name for compatibility."""
    return adaptive_whisper_transcribe(video_path)


# ---------- OCR ----------
# Use the parallel_ocr function directly
from parallel_ocr import parallel_ocr_frames

# Wrapper for compatibility with old code
def ocr_frames(video_path: pathlib.Path) -> str:
    """Process video frames with optimized text detection and OCR."""
    # Use our parallel OCR function
    return parallel_ocr_frames(video_path, max_frames=30)


# ---------- Geocode (Updated) ----------
def geocode_place(place_name: str, genre: str = None, extra_hint: str = None):
    """Geocode a place using Google Maps API with optional genre and extra search hint."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        print("❗ No GOOGLE_API_KEY found in environment.")
        return None

    gm = googlemaps.Client(key)

    query_parts = [place_name]
    if genre:
        query_parts.append(genre)
    if extra_hint:
        query_parts.append(extra_hint)

    query = " ".join(query_parts)

    try:
        res = gm.places(query)
    except Exception as e:
        print(f"❌ Geocoding error for {place_name}: {e}")
        return None

    if not res.get("results"):
        return None

    best = res["results"][0]
    loc = best["geometry"]["location"]

    return {
        "display_address": best.get("formatted_address"),
        "lat": loc["lat"],
        "lon": loc["lng"],
    }


# ---------- Captions ----------
def fetch_caption(url: str) -> str:
    """Return the caption/description text of a Reel, TikTok, or YT Short."""
    ydl_opts = {
        "skip_download": True,  # metadata only
        "quiet": True,
        "simulate": True,
        "forcejson": True,  # return info-dict
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        desc = info.get("description") or ""
        return desc