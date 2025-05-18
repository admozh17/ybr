#!/usr/bin/env python3
"""
Agent Integration for GPT-Centered Video Analysis
This module integrates the GPT-Centered Video Recognition with the agent pipeline.
"""

import os
import sys
import json
import pathlib
import tempfile
import asyncio
import time
from typing import Dict, Any, Optional

# Import from agent module
from agent import (
    run_parallel_tasks,
    fetch_clip,
    parallel_geocode_activities,
    parse_place_info
)

# Import from GPT video recognition module
from gpt_video_recognition import (
    GPTVideoRecognition,
    AnalysisMode,
    analyze_video_sync
)

async def process_video(
    url: str,
    analysis_mode: str = "standard",
    output_path: Optional[str] = None,
    max_frames: int = 5
) -> Dict[str, Any]:
    """
    Process a video using GPT-centered video analysis.
    
    Args:
        url: URL of the video to process
        analysis_mode: Mode for GPT analysis ("basic", "standard", "detailed")
        output_path: Path to save output JSON (optional)
        max_frames: Maximum number of frames to analyze
        
    Returns:
        Dictionary with extraction and recognition results
    """
    start_time = time.time()
    
    # Map string mode to enum
    mode_map = {
        "basic": AnalysisMode.BASIC,
        "standard": AnalysisMode.STANDARD,
        "detailed": AnalysisMode.DETAILED
    }
    
    # Validate mode
    if analysis_mode not in mode_map:
        print(f"⚠️ Invalid analysis mode: {analysis_mode}. Using 'standard'.")
        analysis_mode = "standard"
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        clip_path = tmp_path / "clip.mp4"
        
        try:
            # Download video
            download_start = time.time()
            print(f"📥 Downloading video from {url}...")
            fetch_clip(url, clip_path)
            download_time = time.time() - download_start
            print(f"⏱️ Download completed in {download_time:.2f} seconds")
            
            # Extract content in parallel (speech, OCR, caption)
            extract_start = time.time()
            print("🚀 Starting parallel content extraction...")
            extraction_results = run_parallel_tasks(clip_path, url)
            speech_text = extraction_results.get("speech_text", "")
            frame_text = extraction_results.get("frame_text", "")
            caption_text = extraction_results.get("caption_text", "")
            extraction_time = time.time() - extract_start
            print(f"⏱️ Parallel extraction completed in {extraction_time:.2f} seconds")
            
            # Combine text sources
            fused_text = "\n".join(
                filter(
                    None,
                    [
                        f"SPEECH: {speech_text}" if speech_text else "",
                        f"OCR TEXT: {frame_text}" if frame_text else "",
                        f"CAPTION: {caption_text}" if caption_text else "",
                    ],
                )
            )
            
            print(
                f"🔹 Text fused for analysis ({len(fused_text)} chars):",
                fused_text[:200],
                "..." if len(fused_text) > 200 else "",
            )
            
            # Parse structured info with LLM
            parse_start = time.time()
            print("🧠 Parsing information with LLM...")
            info = parse_place_info(fused_text)
            parse_time = time.time() - parse_start
            print(f"⏱️ Parsing completed in {parse_time:.2f} seconds")
            
            # Run GPT video recognition
            recognition_start = time.time()
            content_type = info.content_type if hasattr(info, 'content_type') else None
            
            print(f"🎬 Starting GPT video analysis (mode: {analysis_mode})...")
            
            # Initialize GPT video recognition system
            recognition = GPTVideoRecognition(
                max_frames=max_frames,
                analysis_mode=mode_map[analysis_mode]
            )
            
            # Run analysis
            recognition_result = analyze_video_sync(
                recognition,
                clip_path,
                content_type=content_type,
                speech_text=speech_text,
                ocr_text=frame_text, 
                caption_text=caption_text
            )
            
            recognition_time = time.time() - recognition_start
            print(f"⏱️ GPT video analysis completed in {recognition_time:.2f} seconds")
            
            # Geocode locations
            geocode_start = time.time()
            print("🌍 Starting parallel geocoding...")
            info.activities = parallel_geocode_activities(info.activities)
            geocode_time = time.time() - geocode_start
            print(f"⏱️ Geocoding completed in {geocode_time:.2f} seconds")
            
            # Build final result
            total_time = time.time() - start_time
            
            # Generate profiling summary
            profile = {
                "download_time": download_time,
                "extraction_time": extraction_time,
                "parse_time": parse_time,
                "recognition_time": recognition_time,
                "geocode_time": geocode_time,
                "total_time": total_time,
            }
            
            # Convert result to dict
            result = info.dict()
            
            # Add raw extraction texts
            result["speech_text"] = speech_text
            result["frame_text"] = frame_text
            result["caption_text"] = caption_text
            
            # Add performance profile
            result["performance_profile"] = profile
            
            # Add GPT recognition data
            result["gpt_analysis"] = recognition_result
            
            # Add activity summary 
            if "activity_summary" in recognition_result:
                result["activity_summary"] = recognition_result["activity_summary"]
            
            # Enhance activities with visual data from GPT-4V
            result = recognition.enhance_extraction_data(recognition_result, result)
            
            # Save result if output path provided
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"✅ Results saved to: {output_path}")
            
            print(f"⏱️ Total processing time: {total_time:.2f} seconds")
            return result
            
        except Exception as e:
            import traceback
            print(f"❌ Error processing video: {str(e)}")
            traceback.print_exc()
            
            error_result = {
                "error": str(e),
                "content_type": "Error", 
                "activities": []
            }
            
            # Save error result if output path provided
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(error_result, f, indent=2, ensure_ascii=False)
                
            return error_result

def main():
    """Command-line interface for GPT-centered video processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process videos with GPT-centered analysis")
    parser.add_argument("--url", required=True, help="URL of video to process")
    parser.add_argument("--out", default="gpt_result.json", help="Output JSON file")
    parser.add_argument("--mode", choices=["basic", "standard", "detailed"], 
                      default="standard", help="Analysis mode")
    parser.add_argument("--frames", type=int, default=5, help="Max frames to analyze")
    parser.add_argument("--time", action="store_true", help="Show detailed timing")
    
    args = parser.parse_args()
    
    try:
        print(f"🎬 Processing video from: {args.url}")
        
        # Process video
        result = asyncio.run(process_video(
            args.url, 
            analysis_mode=args.mode,
            output_path=args.out,
            max_frames=args.frames
        ))
        
        # Check for errors
        if "error" in result:
            print(f"⚠️ Completed with errors. Check {args.out} for details.")
            return
        
        # Print summary
        print(f"✅ Results written to {args.out}")
        
        # Print activity summary
        if "activity_summary" in result:
            print("\n" + "="*60)
            print("GPT VIDEO ANALYSIS")
            print("="*60)
            print(result["activity_summary"])
            print("="*60 + "\n")
        
        # Print performance profile if requested
        if args.time and "performance_profile" in result:
            profile = result["performance_profile"]
            print("\n⏱️ Performance Profile:")
            print(f"  📥 Download: {profile['download_time']:.2f}s")
            print(f"  📝 Content Extraction: {profile['extraction_time']:.2f}s")
            print(f"  🧠 LLM Parsing: {profile['parse_time']:.2f}s")
            print(f"  🎬 GPT Video Analysis: {profile['recognition_time']:.2f}s")
            print(f"  🌍 Geocoding: {profile['geocode_time']:.2f}s")
            print(f"  🏁 Total Time: {profile['total_time']:.2f}s")
            
    except Exception as e:
        print(f"💥 Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Write error information to output file
        error_data = {"error": str(e), "content_type": "Error", "activities": []}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(error_data, f, indent=2, ensure_ascii=False)
            
        print(f"⚠️ Error information written to {args.out}")

# Simple synchronous wrapper for the process_video function
def process_video_sync(url, **kwargs):
    """Synchronous wrapper for process_video"""
    return asyncio.run(process_video(url, **kwargs))

if __name__ == "__main__":
    main()
