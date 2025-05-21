#!/usr/bin/env python3
"""Benchmark script to compare ASR methods."""

import os
import time
import pathlib
import subprocess
import argparse
import json
import tempfile
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# Import necessary functions
from extractor import (
    fetch_clip,
    extract_audio, 
    adaptive_whisper_transcribe, 
    detect_complexity, 
    detect_audio_quality
)

# Define the process_segment function at the top level so it can be pickled
def _process_segment(segment_data_with_params):
    """Process a single audio segment with ASR.
    
    This must be defined at the top level for multiprocessing to work.
    """
    import whisperx
    import torch
    
    segment_idx, segment_path, model_size, device = segment_data_with_params
    print(f"🔊 ASR shard {segment_idx} pid={os.getpid()}")
    
    try:
        # Load appropriate model based on complexity analysis
        if model_size == "large":
            model = whisperx.load_model("large-v2", device=device, compute_type="float16")
        else:
            model = whisperx.load_model("small", device=device, compute_type="int8")
        
        # Run transcription
        result = model.transcribe(segment_path)
        segment_transcript = " ".join(seg["text"] for seg in result["segments"])
        return segment_idx, segment_transcript
    except Exception as e:
        print(f"❌ Error processing segment {segment_idx}: {e}")
        return segment_idx, ""


def whisper_transcribe_segments_forced(clip_path: pathlib.Path, segment_length: float = 8.0) -> str:
    """Modified version of whisper_transcribe_segments that forces segmentation."""
    import whisperx
    import torch
    import numpy as np
    from pydub import AudioSegment
    import time
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    
    print("🎤 Starting forced segmented transcription...")
    start_time = time.time()
    print(f"Main ASR process PID={os.getpid()}")
    
    # Extract audio
    audio_path = extract_audio(clip_path)
    
    # Check duration
    audio = AudioSegment.from_file(audio_path)
    duration_sec = len(audio) / 1000.0
    print(f"Audio duration: {duration_sec:.2f}s")
    
    # Create forced segmentation
    print("🔍 Creating forced segments...")
    
    # Define optimal segment size
    optimal_segment_length = segment_length  # seconds
    
    # Create evenly spaced segments
    segments = []
    num_segments = max(4, int(duration_sec / optimal_segment_length))
    segment_length = duration_sec / num_segments
    
    for i in range(num_segments):
        start = i * segment_length
        end = min(duration_sec, (i + 1) * segment_length)
        segments.append((start, end))
    
    print(f"Created {len(segments)} forced segments of ~{segment_length:.1f}s each")
    
    # Process a sample segment to determine complexity
    sample_segment = segments[0]
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
    
    # Select model size based on complexity
    model_size = "large" if use_large_model else "small"
    print(f"Using {model_size} model based on complexity analysis")
    
    # Prepare segment files for parallel processing
    segment_files = []
    for i, (start_time, end_time) in enumerate(segments):
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
        
        # Include model parameters with each segment
        segment_files.append((i, str(segment_path), model_size, device))
    
    # Process segments in parallel
    all_transcripts = {}
    max_workers = min(multiprocessing.cpu_count() - 1, len(segment_files))
    max_workers = max(1, max_workers)  # Ensure at least 1 worker
    
    print(f"🚀 Processing {len(segment_files)} segments with {max_workers} parallel workers")
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks using the top-level function
            futures = [executor.submit(_process_segment, segment_data) for segment_data in segment_files]
            
            # Process results as they complete
            for future in as_completed(futures):
                try:
                    segment_idx, segment_transcript = future.result()
                    if segment_transcript.strip():
                        all_transcripts[segment_idx] = segment_transcript
                        print(f"✅ Completed segment {segment_idx}")
                except Exception as e:
                    print(f"❌ Error getting result: {e}")
    except Exception as e:
        print(f"❌ Error in parallel processing: {e}")
    
    # If we didn't get any transcripts, use a fallback
    if not all_transcripts:
        print("⚠️ No valid transcripts from parallel processing, falling back to adaptive approach")
        fallback_transcript = adaptive_whisper_transcribe(clip_path)
        end_time = time.time()
        print(f"✅ Fallback transcription completed in {end_time - start_time:.2f} seconds")
        return fallback_transcript

    # Combine all segments in the correct order
    ordered_transcripts = [all_transcripts[idx] for idx in sorted(all_transcripts.keys()) if idx in all_transcripts]
    full_transcript = " ".join(ordered_transcripts)

    end_time = time.time()
    print(f"✅ Forced segmented transcription completed in {end_time - start_time:.2f} seconds")
    print(f"   Used {model_size} model")
    print(f"   Processed {len(all_transcripts)}/{len(segment_files)} segments in parallel")

    return full_transcript


def benchmark_asr_methods(url, output_dir="benchmark_results", num_runs=3, segment_length=8.0):
    """Compare whisper_transcribe_segments vs adaptive_whisper_transcribe performance."""
    # Create output directory
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {
        "video_url": url,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "num_runs": num_runs,
        "segment_length": segment_length,
        "methods": {
            "segmented_forced": {"times": [], "transcript_lengths": []},
            "adaptive": {"times": [], "transcript_lengths": []}
        }
    }
    
    transcripts = {
        "segmented_forced": [],
        "adaptive": []
    }
    
    print(f"Benchmarking ASR methods on video URL: {url}")
    print(f"Running {num_runs} iterations for each method")
    
    # Download the video once to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        clip_path = tmp_path / "clip.mp4"
        
        print(f"Downloading video from URL: {url}")
        fetch_clip(url, clip_path)
        print(f"Video downloaded to: {clip_path}")
        
        # Run multiple times to get average performance
        for i in range(num_runs):
            print(f"\n==== Run {i+1}/{num_runs} ====")
            
            # Test forced segmented approach
            print(f"\n[Testing Forced Segmented Approach]")
            start_time = time.time()
            segmented_transcript = whisper_transcribe_segments_forced(clip_path, segment_length)
            segmented_time = time.time() - start_time
            
            transcripts["segmented_forced"].append(segmented_transcript)
            results["methods"]["segmented_forced"]["times"].append(segmented_time)
            results["methods"]["segmented_forced"]["transcript_lengths"].append(len(segmented_transcript))
            
            print(f"Segmented approach took {segmented_time:.2f} seconds")
            print(f"Transcript length: {len(segmented_transcript)} characters")
            
            # Test adaptive approach
            print(f"\n[Testing Adaptive Approach]")
            start_time = time.time()
            adaptive_transcript = adaptive_whisper_transcribe(clip_path)
            adaptive_time = time.time() - start_time
            
            transcripts["adaptive"].append(adaptive_transcript)
            results["methods"]["adaptive"]["times"].append(adaptive_time)
            results["methods"]["adaptive"]["transcript_lengths"].append(len(adaptive_transcript))
            
            print(f"Adaptive approach took {adaptive_time:.2f} seconds")
            print(f"Transcript length: {len(adaptive_transcript)} characters")
            
            # Calculate similarity between transcripts
            similarity = SequenceMatcher(None, segmented_transcript, adaptive_transcript).ratio()
            print(f"Transcript similarity: {similarity:.4f}")
            
            # Save intermediate results after each run
            with open(output_path / f"run_{i+1}_results.json", "w") as f:
                run_results = {
                    "run": i+1,
                    "segmented_time": segmented_time,
                    "adaptive_time": adaptive_time,
                    "similarity": similarity,
                    "segmented_length": len(segmented_transcript),
                    "adaptive_length": len(adaptive_transcript)
                }
                json.dump(run_results, f, indent=2)
    
    # Calculate averages and summary statistics
    for method in results["methods"]:
        times = results["methods"][method]["times"]
        results["methods"][method]["avg_time"] = sum(times) / len(times)
        results["methods"][method]["min_time"] = min(times)
        results["methods"][method]["max_time"] = max(times)
        
        lengths = results["methods"][method]["transcript_lengths"]
        results["methods"][method]["avg_length"] = sum(lengths) / len(lengths)
    
    # Save transcripts for comparison
    for method in transcripts:
        with open(output_path / f"{method}_transcript.txt", "w") as f:
            f.write(transcripts[method][0])  # Save first run transcript
    
    # Calculate overall transcript similarity
    overall_similarity = SequenceMatcher(
        None, 
        transcripts["segmented_forced"][0], 
        transcripts["adaptive"][0]
    ).ratio()
    results["transcript_similarity"] = overall_similarity
    
    # Save full results
    with open(output_path / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results, transcripts


def visualize_results(results, output_dir="benchmark_results"):
    """Visualize and analyze benchmark results."""
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Extract data for plotting
    methods = list(results["methods"].keys())
    avg_times = [results["methods"][method]["avg_time"] for method in methods]
    min_times = [results["methods"][method]["min_time"] for method in methods]
    max_times = [results["methods"][method]["max_time"] for method in methods]
    
    # Create prettier method names for display
    display_names = {
        "segmented_forced": "Forced Segmented",
        "adaptive": "Adaptive"
    }
    
    # Plot time comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        [display_names.get(m, m) for m in methods], 
        avg_times, 
        color=['steelblue', 'darkorange']
    )
    
    # Add min/max as error bars
    for i, (bar, min_t, max_t) in enumerate(zip(bars, min_times, max_times)):
        plt.plot([bar.get_x() + bar.get_width()/2, bar.get_x() + bar.get_width()/2], 
                 [min_t, max_t], 'k-', lw=1.5)
        plt.plot([bar.get_x() + bar.get_width()/2 - 0.1, bar.get_x() + bar.get_width()/2 + 0.1], 
                 [min_t, min_t], 'k-', lw=1.5)
        plt.plot([bar.get_x() + bar.get_width()/2 - 0.1, bar.get_x() + bar.get_width()/2 + 0.1], 
                 [max_t, max_t], 'k-', lw=1.5)
    
    plt.title('ASR Processing Time Comparison', fontsize=16)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(avg_times):
        plt.text(i, v + 0.5, f"{v:.2f}s", ha='center', fontsize=12)
    
    plt.savefig(output_path / 'asr_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary text report
    with open(output_path / "summary_report.txt", "w") as f:
        f.write("===== ASR BENCHMARK SUMMARY =====\n\n")
        f.write(f"Video URL: {results['video_url']}\n")
        f.write(f"Date: {results['timestamp']}\n")
        f.write(f"Runs: {results['num_runs']}\n")
        f.write(f"Segment Length: {results.get('segment_length', 8.0)}s\n\n")
        
        f.write("===== PERFORMANCE COMPARISON =====\n\n")
        for method in methods:
            display_name = display_names.get(method, method)
            f.write(f"{display_name}:\n")
            f.write(f"  Average Time: {results['methods'][method]['avg_time']:.2f} seconds\n")
            f.write(f"  Range: {results['methods'][method]['min_time']:.2f}s - {results['methods'][method]['max_time']:.2f}s\n")
            f.write(f"  Average Transcript Length: {results['methods'][method]['avg_length']} characters\n\n")
            
        f.write(f"Transcript Similarity: {results.get('transcript_similarity', 0):.4f}\n\n")
        
        # Calculate speedup
        if "adaptive" in results["methods"] and "segmented_forced" in results["methods"]:
            adaptive_time = results["methods"]["adaptive"]["avg_time"]
            segmented_time = results["methods"]["segmented_forced"]["avg_time"]
            speedup = adaptive_time / segmented_time if segmented_time > 0 else 0
            
            f.write(f"Speedup: {speedup:.2f}x\n")
            if speedup > 1:
                f.write(f"Forced Segmented approach is {speedup:.2f}x faster than Adaptive approach\n")
            else:
                f.write(f"Adaptive approach is {1/speedup:.2f}x faster than Forced Segmented approach\n")
    
    print(f"Results saved to {output_path}")
    print(f"Check {output_path}/summary_report.txt for detailed results")


def main():
    """Run the ASR benchmark."""
    parser = argparse.ArgumentParser(description="ASR Performance Benchmark")
    parser.add_argument("url", help="URL of the video to analyze")
    parser.add_argument("--output", "-o", default="benchmark_results", help="Output directory for results")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--segment-length", "-s", type=float, default=8.0, 
                       help="Segment length in seconds for forced segmentation")
    args = parser.parse_args()
    
    results, transcripts = benchmark_asr_methods(
        args.url, 
        output_dir=args.output, 
        num_runs=args.runs,
        segment_length=args.segment_length
    )
    
    visualize_results(results, output_dir=args.output)
    
    # Print summary
    for method in results["methods"]:
        avg_time = results["methods"][method]["avg_time"]
        print(f"{method}: {avg_time:.2f}s")
    
    print(f"Transcript similarity: {results.get('transcript_similarity', 0):.4f}")
    
    # Recommend the faster method
    if results["methods"]["segmented_forced"]["avg_time"] < results["methods"]["adaptive"]["avg_time"]:
        speedup = results["methods"]["adaptive"]["avg_time"] / results["methods"]["segmented_forced"]["avg_time"]
        print(f"Recommendation: Use forced segmentation for {speedup:.2f}x speedup")
    else:
        speedup = results["methods"]["segmented_forced"]["avg_time"] / results["methods"]["adaptive"]["avg_time"]
        print(f"Recommendation: Stick with adaptive approach (segmented is {speedup:.2f}x slower)")


if __name__ == "__main__":
    main()