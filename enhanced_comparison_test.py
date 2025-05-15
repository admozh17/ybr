#!/usr/bin/env python3
"""
Comprehensive test comparing all Whisper models with and without audio segmentation.
Saves raw output files for manual comparison and shows detailed timing.
"""

import argparse
import json
import pathlib
import time
import tempfile
import os
import sys
import datetime
from pydub import AudioSegment
import subprocess
import torch
import whisperx
import matplotlib.pyplot as plt
import numpy as np

# Import original functions
from extractor import (
    fetch_clip,
    whisper_transcribe,
    ocr_frames,
    geocode_place,
    fetch_caption,
    extract_audio,
    detect_speech_segments,
)

# Import LLM parser
from llm_parser import parse_place_info


# Create directory for outputs
def setup_output_dirs(base_path):
    """Create directories for various outputs."""
    # Main directories
    raw_dir = base_path / "raw_outputs"
    analysis_dir = base_path / "analysis"

    raw_dir.mkdir(exist_ok=True, parents=True)
    analysis_dir.mkdir(exist_ok=True, parents=True)

    return raw_dir, analysis_dir


def save_output(output_dir, name, content):
    """Save output to file."""
    with open(output_dir / f"{name}.txt", "w") as f:
        f.write(content)


# Transcription using large model without segmentation
def run_large_model(video_path, output_dir):
    """Run transcription using large model without segmentation."""
    print("\n=== RUNNING LARGE MODEL (WITHOUT SEGMENTATION) ===")
    start_time = time.time()

    # Extract audio
    audio_path = extract_audio(video_path)

    # Use large model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model(
        "large-v2",
        device=device,
        compute_type="float32",  # Use float32 for compatibility
    )

    # Transcribe
    result = model.transcribe(str(audio_path))

    # Extract text
    text = " ".join(seg["text"] for seg in result["segments"])
    segments = result["segments"]

    end_time = time.time()
    duration = end_time - start_time

    print(f"Large model transcription completed in {duration:.2f} seconds")
    print(f"Extracted {len(text)} characters, {len(text.split())} words")

    # Save raw output
    save_output(output_dir, "large_model_output", text)

    # Save segments
    with open(output_dir / "large_model_segments.json", "w") as f:
        json.dump(segments, f, indent=2)

    return text, segments, duration


# Transcription using small model without segmentation
def run_small_model(video_path, output_dir):
    """Run transcription using small model without segmentation."""
    print("\n=== RUNNING SMALL MODEL (WITHOUT SEGMENTATION) ===")
    start_time = time.time()

    # Extract audio
    audio_path = extract_audio(video_path)

    # Use small model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model("small", device=device, compute_type="int8")

    # Transcribe
    result = model.transcribe(str(audio_path))

    # Extract text
    text = " ".join(seg["text"] for seg in result["segments"])
    segments = result["segments"]

    end_time = time.time()
    duration = end_time - start_time

    print(f"Small model transcription completed in {duration:.2f} seconds")
    print(f"Extracted {len(text)} characters, {len(text.split())} words")

    # Save raw output
    save_output(output_dir, "small_model_output", text)

    # Save segments
    with open(output_dir / "small_model_segments.json", "w") as f:
        json.dump(segments, f, indent=2)

    return text, segments, duration


# Transcription using adaptive model with segmentation
def run_adaptive_model(video_path, output_dir):
    """Run transcription using adaptive model with segmentation."""
    print("\n=== RUNNING ADAPTIVE MODEL (WITH SEGMENTATION) ===")
    start_time = time.time()

    # Use the existing whisper_transcribe function which should be the adaptive implementation
    text = whisper_transcribe(video_path)

    end_time = time.time()
    duration = end_time - start_time

    print(f"Adaptive model transcription completed in {duration:.2f} seconds")
    print(f"Extracted {len(text)} characters, {len(text.split())} words")

    # Save raw output
    save_output(output_dir, "adaptive_model_output", text)

    return text, duration


# OCR using new implementation
def run_new_ocr(video_path, output_dir):
    """Run OCR using new implementation."""
    print("\n=== RUNNING NEW OCR ===")
    start_time = time.time()

    # Use existing OCR function which should be the new implementation
    text = ocr_frames(video_path)

    end_time = time.time()
    duration = end_time - start_time

    print(f"New OCR completed in {duration:.2f} seconds")
    print(f"Extracted {len(text)} characters, {len(text.split())} words")

    # Save raw output
    save_output(output_dir, "new_ocr_output", text)

    return text, duration


# Caption extraction
def run_caption_extraction(url, output_dir):
    """Run caption extraction."""
    print("\n=== RUNNING CAPTION EXTRACTION ===")
    start_time = time.time()

    # Fetch caption
    text = fetch_caption(url)

    end_time = time.time()
    duration = end_time - start_time

    print(f"Caption extraction completed in {duration:.2f} seconds")
    print(f"Extracted {len(text)} characters, {len(text.split())} words")

    # Save raw output
    save_output(output_dir, "caption_output", text)

    return text, duration


# Analyze audio for speech segments
def analyze_audio(video_path, output_dir):
    """Analyze audio to detect speech segments."""
    print("\n=== ANALYZING AUDIO ===")

    # Extract audio
    audio_path = extract_audio(video_path)

    # Get audio duration
    audio = AudioSegment.from_file(audio_path)
    duration_sec = len(audio) / 1000.0

    print(f"Total audio duration: {duration_sec:.2f} seconds")

    # Detect speech segments
    segments = detect_speech_segments(audio_path)

    # Calculate speech duration
    speech_duration = sum(end - start for start, end in segments)

    # Calculate percentage
    speech_percentage = (
        (speech_duration / duration_sec) * 100 if duration_sec > 0 else 0
    )

    print(f"Detected {len(segments)} speech segments")
    print(f"Speech duration: {speech_duration:.2f} seconds ({speech_percentage:.1f}%)")
    print(
        f"Non-speech duration: {duration_sec - speech_duration:.2f} seconds ({100 - speech_percentage:.1f}%)"
    )

    # Save segment information
    segment_info = {
        "total_duration": duration_sec,
        "speech_duration": speech_duration,
        "speech_percentage": speech_percentage,
        "num_segments": len(segments),
        "segments": [
            {"start": start, "end": end, "duration": end - start}
            for start, end in segments
        ],
    }

    with open(output_dir / "speech_segments.json", "w") as f:
        json.dump(segment_info, f, indent=2)

    return segment_info


# Compare model outputs
def compare_transcriptions(large_text, small_text, adaptive_text, output_dir):
    """Compare and analyze different transcription outputs."""
    print("\n=== COMPARING TRANSCRIPTIONS ===")

    # Tokenize to words
    large_words = large_text.lower().split()
    small_words = small_text.lower().split()
    adaptive_words = adaptive_text.lower().split()

    # Get unique words
    large_unique = set(large_words)
    small_unique = set(small_words)
    adaptive_unique = set(adaptive_words)

    # Calculate common words
    large_small_common = large_unique.intersection(small_unique)
    large_adaptive_common = large_unique.intersection(adaptive_unique)
    small_adaptive_common = small_unique.intersection(adaptive_unique)

    # Calculate similarity percentages
    large_small_similarity = (
        len(large_small_common) / len(large_unique.union(small_unique))
        if large_unique or small_unique
        else 0
    )
    large_adaptive_similarity = (
        len(large_adaptive_common) / len(large_unique.union(adaptive_unique))
        if large_unique or adaptive_unique
        else 0
    )
    small_adaptive_similarity = (
        len(small_adaptive_common) / len(small_unique.union(adaptive_unique))
        if small_unique or adaptive_unique
        else 0
    )

    print(f"Large model: {len(large_words)} words, {len(large_unique)} unique")
    print(f"Small model: {len(small_words)} words, {len(small_unique)} unique")
    print(f"Adaptive model: {len(adaptive_words)} words, {len(adaptive_unique)} unique")

    print(f"Large-Small similarity: {large_small_similarity:.2f}")
    print(f"Large-Adaptive similarity: {large_adaptive_similarity:.2f}")
    print(f"Small-Adaptive similarity: {small_adaptive_similarity:.2f}")

    # Generate word cloud comparison (difference between models)
    large_not_in_adaptive = large_unique - adaptive_unique
    adaptive_not_in_large = adaptive_unique - large_unique

    small_not_in_adaptive = small_unique - adaptive_unique
    adaptive_not_in_small = adaptive_unique - small_unique

    # Save unique words to each model
    with open(output_dir / "large_unique_words.txt", "w") as f:
        f.write("\n".join(sorted(large_not_in_adaptive)))

    with open(output_dir / "adaptive_unique_words.txt", "w") as f:
        f.write("\n".join(sorted(adaptive_not_in_large)))

    with open(output_dir / "small_unique_words.txt", "w") as f:
        f.write("\n".join(sorted(small_not_in_adaptive)))

    # Save comparison results
    comparison = {
        "word_counts": {
            "large": len(large_words),
            "small": len(small_words),
            "adaptive": len(adaptive_words),
        },
        "unique_words": {
            "large": len(large_unique),
            "small": len(small_unique),
            "adaptive": len(adaptive_unique),
        },
        "similarity": {
            "large_small": large_small_similarity,
            "large_adaptive": large_adaptive_similarity,
            "small_adaptive": small_adaptive_similarity,
        },
        "unique_to_model": {
            "large_not_in_adaptive": len(large_not_in_adaptive),
            "adaptive_not_in_large": len(adaptive_not_in_large),
            "small_not_in_adaptive": len(small_not_in_adaptive),
            "adaptive_not_in_small": len(adaptive_not_in_small),
        },
    }

    with open(output_dir / "transcription_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    return comparison


# Create comparison visuals
def create_visualizations(
    segment_info, large_segments, comparison, timings, output_dir
):
    """Create visualizations of the analysis."""
    analysis_dir = output_dir / "visualizations"
    analysis_dir.mkdir(exist_ok=True)

    # 1. Audio segmentation timeline
    plt.figure(figsize=(12, 6))

    # Plot full audio duration
    plt.barh(0, segment_info["total_duration"], color="lightgray", height=0.5)

    # Plot speech segments
    for i, seg in enumerate(segment_info["segments"]):
        plt.barh(0, seg["duration"], left=seg["start"], color="blue", alpha=0.6)

    plt.yticks([0], ["Audio"])
    plt.xlabel("Time (seconds)")
    plt.title("Speech Segments in Audio")
    plt.tight_layout()
    plt.savefig(analysis_dir / "speech_segments.png")
    plt.close()

    # 2. Performance comparison
    plt.figure(figsize=(10, 6))
    models = ["Large", "Small", "Adaptive"]

    # Timing data
    timing_data = [timings["large"], timings["small"], timings["adaptive"]]
    plt.bar(models, timing_data, color=["red", "green", "blue"])

    plt.ylabel("Time (seconds)")
    plt.title("Transcription Time by Model")

    # Add timing labels
    for i, v in enumerate(timing_data):
        plt.text(i, v + 1, f"{v:.1f}s", ha="center")

    plt.tight_layout()
    plt.savefig(analysis_dir / "model_timing.png")
    plt.close()

    # 3. Word count comparison
    plt.figure(figsize=(10, 6))
    word_counts = [
        comparison["word_counts"]["large"],
        comparison["word_counts"]["small"],
        comparison["word_counts"]["adaptive"],
    ]

    plt.bar(models, word_counts, color=["red", "green", "blue"])
    plt.ylabel("Word Count")
    plt.title("Words Extracted by Model")

    # Add word count labels
    for i, v in enumerate(word_counts):
        plt.text(i, v + 5, str(v), ha="center")

    plt.tight_layout()
    plt.savefig(analysis_dir / "word_counts.png")
    plt.close()

    print(f"Visualizations saved to {analysis_dir}")


# Run the full comparison test
def run_comparison_test(url, output_dir):
    """Run comprehensive comparison of different transcription approaches."""
    print(f"\n{'=' * 60}")
    print(f"Starting comprehensive comparison for URL: {url}")
    print(f"{'=' * 60}\n")

    # Create output directory
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Setup directories
    raw_dir, analysis_dir = setup_output_dirs(output_path)

    # Download video
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        clip_path = tmp_path / "clip.mp4"

        print("Downloading video...")
        fetch_clip(url, clip_path)

        # Analyze audio segments
        segment_info = analyze_audio(clip_path, analysis_dir)

        # Run each model
        large_text, large_segments, large_time = run_large_model(clip_path, raw_dir)
        small_text, small_segments, small_time = run_small_model(clip_path, raw_dir)
        adaptive_text, adaptive_time = run_adaptive_model(clip_path, raw_dir)

        # Run OCR
        ocr_text, ocr_time = run_new_ocr(clip_path, raw_dir)

        # Run caption extraction
        caption_text, caption_time = run_caption_extraction(url, raw_dir)

    # Compare transcriptions
    comparison = compare_transcriptions(
        large_text, small_text, adaptive_text, analysis_dir
    )

    # Create visualizations
    timings = {"large": large_time, "small": small_time, "adaptive": adaptive_time}

    create_visualizations(
        segment_info, large_segments, comparison, timings, analysis_dir
    )

    # Generate comparison report
    generate_comparison_report(
        large_text,
        small_text,
        adaptive_text,
        ocr_text,
        caption_text,
        segment_info,
        comparison,
        timings,
        output_path,
    )

    print(f"\n{'=' * 60}")
    print(f"Comparison test completed")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 60}")

    # Print key timings
    print("\nPERFORMANCE SUMMARY:")
    print(f"Large model: {large_time:.2f} seconds")
    print(f"Small model: {small_time:.2f} seconds")
    print(f"Adaptive model: {adaptive_time:.2f} seconds")
    print(f"OCR: {ocr_time:.2f} seconds")

    # Print word counts
    print("\nEXTRACTED CONTENT SUMMARY:")
    print(f"Large model: {len(large_text.split())} words")
    print(f"Small model: {len(small_text.split())} words")
    print(f"Adaptive model: {len(adaptive_text.split())} words")
    print(f"OCR: {len(ocr_text.split())} words")

    # Print audio segmentation
    print("\nAUDIO SEGMENTATION:")
    print(f"Total audio: {segment_info['total_duration']:.2f} seconds")
    print(
        f"Speech content: {segment_info['speech_duration']:.2f} seconds ({segment_info['speech_percentage']:.1f}%)"
    )
    print(f"Speech segments: {segment_info['num_segments']}")

    print("\nSee detailed reports and visualizations in the output directory.")


def generate_comparison_report(
    large_text,
    small_text,
    adaptive_text,
    ocr_text,
    caption_text,
    segment_info,
    comparison,
    timings,
    output_path,
):
    """Generate a comprehensive comparison report."""
    with open(output_path / "comprehensive_report.txt", "w") as f:
        f.write("COMPREHENSIVE MODEL COMPARISON REPORT\n")
        f.write("===================================\n\n")
        f.write(
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        # Audio segmentation analysis
        f.write("AUDIO SEGMENTATION ANALYSIS\n")
        f.write("--------------------------\n")
        f.write(f"Total audio duration: {segment_info['total_duration']:.2f} seconds\n")
        f.write(
            f"Speech content: {segment_info['speech_duration']:.2f} seconds ({segment_info['speech_percentage']:.1f}%)\n"
        )
        f.write(
            f"Non-speech content: {segment_info['total_duration'] - segment_info['speech_duration']:.2f} seconds ({100 - segment_info['speech_percentage']:.1f}%)\n"
        )
        f.write(f"Number of speech segments: {segment_info['num_segments']}\n\n")

        f.write("Speech segments:\n")
        for i, segment in enumerate(segment_info["segments"]):
            f.write(
                f"  Segment {i + 1}: {segment['start']:.2f}s - {segment['end']:.2f}s (duration: {segment['duration']:.2f}s)\n"
            )

        # Performance comparison
        f.write("\n\nPERFORMANCE COMPARISON\n")
        f.write("---------------------\n")
        f.write(f"Large model (without segmentation): {timings['large']:.2f} seconds\n")
        f.write(f"Small model (without segmentation): {timings['small']:.2f} seconds\n")
        f.write(
            f"Adaptive model (with segmentation): {timings['adaptive']:.2f} seconds\n\n"
        )

        # Speedup calculations
        large_to_adaptive = (
            (timings["large"] / timings["adaptive"]) if timings["adaptive"] > 0 else 0
        )
        small_to_adaptive = (
            (timings["small"] / timings["adaptive"]) if timings["adaptive"] > 0 else 0
        )

        f.write(f"Speedup: Large → Adaptive: {large_to_adaptive:.2f}x\n")
        f.write(f"Speedup: Small → Adaptive: {small_to_adaptive:.2f}x\n")

        # Content comparison
        f.write("\n\nCONTENT COMPARISON\n")
        f.write("-----------------\n")

        wc = comparison["word_counts"]
        uw = comparison["unique_words"]

        f.write(f"Large model: {wc['large']} words, {uw['large']} unique words\n")
        f.write(f"Small model: {wc['small']} words, {uw['small']} unique words\n")
        f.write(
            f"Adaptive model: {wc['adaptive']} words, {uw['adaptive']} unique words\n\n"
        )

        sim = comparison["similarity"]
        f.write(f"Text similarity (Jaccard index):\n")
        f.write(f"  Large-Small: {sim['large_small']:.2f}\n")
        f.write(f"  Large-Adaptive: {sim['large_adaptive']:.2f}\n")
        f.write(f"  Small-Adaptive: {sim['small_adaptive']:.2f}\n\n")

        uniq = comparison["unique_to_model"]
        f.write(f"Words unique to each model:\n")
        f.write(
            f"  Words in Large but not in Adaptive: {uniq['large_not_in_adaptive']}\n"
        )
        f.write(
            f"  Words in Adaptive but not in Large: {uniq['adaptive_not_in_large']}\n"
        )
        f.write(
            f"  Words in Small but not in Adaptive: {uniq['small_not_in_adaptive']}\n"
        )
        f.write(
            f"  Words in Adaptive but not in Small: {uniq['adaptive_not_in_small']}\n"
        )

        # Text samples
        f.write("\n\nTEXT SAMPLES\n")
        f.write("-----------\n")

        f.write("\nLarge model transcription (first 300 chars):\n")
        f.write(large_text[:300] + "...\n")

        f.write("\nSmall model transcription (first 300 chars):\n")
        f.write(small_text[:300] + "...\n")

        f.write("\nAdaptive model transcription (first 300 chars):\n")
        f.write(adaptive_text[:300] + "...\n")

        f.write("\nOCR text (first 300 chars):\n")
        f.write(ocr_text[:300] + "...\n")

        f.write("\nCaption text (first 300 chars):\n")
        f.write(caption_text[:300] + "...\n")

        # Files saved
        f.write("\n\nFILES SAVED\n")
        f.write("-----------\n")
        f.write("Raw output files:\n")
        f.write("  - raw_outputs/large_model_output.txt (Large model transcription)\n")
        f.write("  - raw_outputs/small_model_output.txt (Small model transcription)\n")
        f.write(
            "  - raw_outputs/adaptive_model_output.txt (Adaptive model transcription)\n"
        )
        f.write("  - raw_outputs/new_ocr_output.txt (OCR text)\n")
        f.write("  - raw_outputs/caption_output.txt (Caption text)\n\n")

        f.write("Analysis files:\n")
        f.write("  - analysis/speech_segments.json (Audio segmentation details)\n")
        f.write(
            "  - analysis/transcription_comparison.json (Detailed comparison metrics)\n"
        )
        f.write("  - analysis/large_unique_words.txt (Words unique to Large model)\n")
        f.write(
            "  - analysis/adaptive_unique_words.txt (Words unique to Adaptive model)\n"
        )
        f.write("  - analysis/small_unique_words.txt (Words unique to Small model)\n\n")

        f.write("Visualizations:\n")
        f.write(
            "  - analysis/visualizations/speech_segments.png (Audio segmentation timeline)\n"
        )
        f.write(
            "  - analysis/visualizations/model_timing.png (Transcription time comparison)\n"
        )
        f.write("  - analysis/visualizations/word_counts.png (Word count comparison)\n")

        f.write("\n\nANALYSIS CONCLUSION\n")
        f.write("-----------------\n")
        if large_to_adaptive > 1.5:
            f.write(
                f"The adaptive model with segmentation is significantly faster ({large_to_adaptive:.2f}x) than the large model.\n"
            )
        elif large_to_adaptive > 1.0:
            f.write(
                f"The adaptive model with segmentation is somewhat faster ({large_to_adaptive:.2f}x) than the large model.\n"
            )
        else:
            f.write(
                f"The adaptive model with segmentation is not faster than the large model in this case.\n"
            )

        if sim["large_adaptive"] > 0.9:
            f.write(
                "The content extracted by the adaptive model is very similar to the large model.\n"
            )
        elif sim["large_adaptive"] > 0.7:
            f.write(
                "The content extracted by the adaptive model is moderately similar to the large model.\n"
            )
        else:
            f.write(
                "There are significant differences between the content extracted by the adaptive and large models.\n"
            )

        f.write(
            "\nFor detailed word-by-word comparison, examine the raw output files.\n"
        )


def main():
    ap = argparse.ArgumentParser(description="Comprehensive model comparison test")
    ap.add_argument("--url", required=True, help="Video URL to test")
    ap.add_argument("--out", default="model_comparison", help="Output directory")
    args = ap.parse_args()

    run_comparison_test(args.url, args.out)


if __name__ == "__main__":
    main()
