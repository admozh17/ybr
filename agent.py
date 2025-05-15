#!/usr/bin/env python3
"""High‑level orchestration of the Info‑Extractor Agent with parallel processing."""

import argparse
import json
import pathlib
import tempfile
import traceback
import concurrent.futures
import time
from typing import Dict, Any, List, Tuple
from extractor import (
    fetch_clip,
    fetch_caption,
    whisper_transcribe,
    ocr_frames,
    geocode_place,
)
from llm_parser import parse_place_info


def run_parallel_tasks(video_path: pathlib.Path, url: str) -> Dict[str, Any]:
    """Run speech, OCR, and caption extraction in parallel."""
    results = {}

    # Create a thread pool for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks to run in parallel
        speech_future = executor.submit(whisper_transcribe, video_path)
        ocr_future = executor.submit(ocr_frames, video_path)
        caption_future = executor.submit(fetch_caption, url)

        # Monitor progress
        pending = [speech_future, ocr_future, caption_future]
        task_names = ["Speech transcription", "OCR extraction", "Caption fetching"]

        # Wait for all tasks to complete or handle timeouts/errors
        for future, task_name in zip(
            concurrent.futures.as_completed(pending), task_names
        ):
            try:
                if future == speech_future:
                    results["speech_text"] = future.result()
                    print(f"✅ {task_name} completed")
                elif future == ocr_future:
                    results["frame_text"] = future.result()
                    print(f"✅ {task_name} completed")
                elif future == caption_future:
                    results["caption_text"] = future.result()
                    print(f"✅ {task_name} completed")
            except Exception as e:
                print(f"❌ {task_name} failed: {str(e)}")
                if future == speech_future:
                    results["speech_text"] = ""
                elif future == ocr_future:
                    results["frame_text"] = ""
                elif future == caption_future:
                    results["caption_text"] = ""

    return results


def parallel_geocode_activities(activities):
    """Process geocoding for multiple activities in parallel."""
    if not activities:
        return activities

    # Prepare geocoding tasks
    geocode_tasks = []

    for i, activity in enumerate(activities):
        if not activity.place_name:
            continue

        # Build hint per activity
        hints = []
        if activity.availability.city:
            hints.append(activity.availability.city)
        if activity.availability.state:
            hints.append(activity.availability.state)
        if activity.availability.country:
            hints.append(activity.availability.country)
        extra_hint = ", ".join(hints) if hints else None

        # Skip if already has street address
        if not activity.availability.street_address:
            geocode_tasks.append(
                {
                    "index": i,
                    "place_name": activity.place_name,
                    "genre": activity.genre,
                    "extra_hint": extra_hint,
                }
            )

    # Run geocoding tasks in parallel
    if geocode_tasks:
        print(f"🌍 Geocoding {len(geocode_tasks)} locations in parallel...")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(10, len(geocode_tasks))
        ) as executor:
            # Submit all geocoding tasks
            future_to_task = {
                executor.submit(
                    geocode_place, task["place_name"], task["genre"], task["extra_hint"]
                ): task
                for task in geocode_tasks
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                activity_index = task["index"]

                try:
                    geo = future.result()

                    if geo:
                        print(f"✅ Geocode success for: {task['place_name']}")
                        # Update activity with geocode results
                        activities[
                            activity_index
                        ].availability.street_address = geo.get("display_address")

                        # Only update these fields if they're empty
                        if not activities[activity_index].availability.city:
                            activities[activity_index].availability.city = geo.get(
                                "city"
                            )
                        if not activities[activity_index].availability.state:
                            activities[activity_index].availability.state = geo.get(
                                "state"
                            )
                        if not activities[activity_index].availability.country:
                            activities[activity_index].availability.country = geo.get(
                                "country"
                            )
                        if not activities[activity_index].availability.region:
                            activities[activity_index].availability.region = geo.get(
                                "region"
                            )
                    else:
                        print(f"❌ Geocode failed for: {task['place_name']}")
                except Exception as e:
                    print(f"❌ Geocode error for {task['place_name']}: {str(e)}")

    return activities


def run(url: str):
    """Process a single short‑form video URL and return parsed info with parallel processing."""
    start_time = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        clip_path = tmp_path / "clip.mp4"

        try:
            # Download video (this step must run first)
            download_start = time.time()
            print(f"📥 Downloading video from {url}...")
            fetch_clip(url, clip_path)
            download_time = time.time() - download_start
            print(f"⏱️ Download completed in {download_time:.2f} seconds")

            # --- Step 1: Extract raw content in parallel ---
            extract_start = time.time()
            print("🚀 Starting parallel content extraction...")

            extraction_results = run_parallel_tasks(clip_path, url)
            speech_text = extraction_results.get("speech_text", "")
            frame_text = extraction_results.get("frame_text", "")
            caption_text = extraction_results.get("caption_text", "")

            extraction_time = time.time() - extract_start
            print(f"⏱️ Parallel extraction completed in {extraction_time:.2f} seconds")

            # --- Step 2: Combine text sources ---
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
                f"🔹 Text fused for LLM ({len(fused_text)} chars):",
                fused_text[:200],
                "..." if len(fused_text) > 200 else "",
            )

            # --- Step 3: Parse structured info from text ---
            parse_start = time.time()
            print("🧠 Parsing information with LLM...")
            info = parse_place_info(fused_text)
            parse_time = time.time() - parse_start
            print(f"⏱️ Parsing completed in {parse_time:.2f} seconds")

            # --- Step 4: Geocode missing locations in parallel ---
            geocode_start = time.time()
            print("🌍 Starting parallel geocoding...")
            info.activities = parallel_geocode_activities(info.activities)
            geocode_time = time.time() - geocode_start
            print(f"⏱️ Geocoding completed in {geocode_time:.2f} seconds")

            total_time = time.time() - start_time
            print(f"⏱️ Total processing time: {total_time:.2f} seconds")

            # Generate profiling summary
            profile = {
                "download_time": download_time,
                "extraction_time": extraction_time,
                "parse_time": parse_time,
                "geocode_time": geocode_time,
                "total_time": total_time,
            }

            result = info.dict()
            result["performance_profile"] = profile

            return result
        except Exception as e:
            print(f"❌ Error processing video: {str(e)}")
            traceback.print_exc()
            return {"error": str(e), "content_type": "Error", "activities": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Reel / TikTok / Short URL")
    ap.add_argument("--out", default="result.json")
    ap.add_argument(
        "--time", action="store_true", help="Show detailed timing information"
    )
    args = ap.parse_args()

    try:
        start_time = time.time()
        print(f"🎬 Processing video from: {args.url}")
        data = run(args.url)

        # Ensure we write some valid JSON even if there was an error
        pathlib.Path(args.out).write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )

        if "error" in data:
            print(f"⚠️ Completed with errors. Check {args.out} for details.")
        else:
            print(f"✅ Parsed info written to {args.out}")

            # Print a summary of what was found
            num_activities = len(data.get("activities", []))
            print(f"📊 Found {num_activities} activities/places")

            for i, activity in enumerate(data.get("activities", [])):
                print(
                    f"  🏙️ {i + 1}. {activity.get('place_name', 'Unknown')} ({activity.get('genre', 'Unknown')})"
                )

            # Print performance profile if requested
            if args.time and "performance_profile" in data:
                profile = data["performance_profile"]
                print("\n⏱️ Performance Profile:")
                print(f"  📥 Download: {profile['download_time']:.2f}s")
                print(f"  📝 Content Extraction: {profile['extraction_time']:.2f}s")
                print(f"  🧠 LLM Parsing: {profile['parse_time']:.2f}s")
                print(f"  🌍 Geocoding: {profile['geocode_time']:.2f}s")
                print(f"  🏁 Total Time: {profile['total_time']:.2f}s")

    except Exception as e:
        print(f"💥 Critical error: {str(e)}")
        traceback.print_exc()

        # Write error information to output file
        error_data = {"error": str(e), "content_type": "Error", "activities": []}
        pathlib.Path(args.out).write_text(
            json.dumps(error_data, indent=2, ensure_ascii=False)
        )
        print(f"⚠️ Error information written to {args.out}")


if __name__ == "__main__":
    main()
