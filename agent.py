#!/usr/bin/env python3
"""High‑level orchestration of the Info‑Extractor Agent with waterfall pipeline."""

import argparse
import json
import pathlib
import tempfile
import traceback
import concurrent.futures
import time
import asyncio
import os
import aiohttp
import re
import base64
import io
from typing import Dict, Any, List, Tuple, Optional, Set
from functools import partial
from PIL import Image
import multiprocessing


# ── Core extraction utilities ───────────────────────────────────────────
from extractor import (
    fetch_clip,
    fetch_caption,
    extract_audio,
    detect_speech_segments,
    ocr_frames,
    geocode_place,
    adaptive_whisper_transcribe
)

# parallel OCR helper now lives in its own file
from parallel_ocr import parallel_ocr_frames

# Use our local fix instead of the missing function
from extractor import whisper_transcribe_segments




from llm_parser import parse_place_info
from clip_gate import ClipGate

# Video recognition imports for frame extraction
import cv2
import numpy as np
import torch
from enum import Enum

# Import OpenAI for GPT-Vision analysis
import openai




# Constants and environment configuration
BACKEND_MODE = os.environ.get("BACKEND_MODE", "local")
ASR_URL = os.environ.get("ASR_URL", "http://127.0.0.1:8002/asr")
VISION_URL = os.environ.get("VISION_URL", "http://127.0.0.1:8001/infer")
S3_BUCKET = os.environ.get("S3_BUCKET", "video-analysis-dev")

# Set up OpenAI client for GPT-Vision
openai_client = openai.OpenAI()


class PipelineStage(Enum):
    """Enumeration of pipeline stages."""
    STAGE_0_METADATA = 0  # Page meta, caption track (instant)
    STAGE_1_BASIC = 1     # VAD-sharded Whisper + regex NER + CLIP gate
    STAGE_2_VISION = 2    # GPT-Vision analysis on gated frames + food analysis
    STAGE_3_FUSION = 3    # Single GPT-4o fusion → DB → UI


class ExtractionState:
    """Object tracking the current state of extraction."""
    
    def __init__(self):
        self.url = ""
        self.started_at = time.time()
        self.completed_stages = set()
        self.caption_text = ""
        self.speech_text = ""
        self.frame_text = ""
        self.frames = []
        self.gated_frames = []
        self.gated_frame_indices = []
        self.visual_results = []
        self.entities = []
        self.clip_path = None
        self.current_stage = PipelineStage.STAGE_0_METADATA
        self.parsed_info = None
        self.performance_profile = {}
        self.error = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a dictionary."""
        result = {
            "url": self.url,
            "processing_time": time.time() - self.started_at,
            "completed_stages": [stage.value for stage in self.completed_stages],
            "current_stage": self.current_stage.value,
            "caption_text": self.caption_text,
            "speech_text": self.speech_text,
            "frame_text": self.frame_text,
            "num_frames": len(self.frames),
            "num_gated_frames": len(self.gated_frames),
            "performance_profile": self.performance_profile,
        }
        
        # Add parsed info if available
        if self.parsed_info:
            result.update(self.parsed_info)
        
        # Add error if present
        if self.error:
            result["error"] = str(self.error)
        
        return result


class Agent:
    """Pipeline orchestrator implementing waterfall architecture."""
    
    def __init__(self, use_gpu: bool = True):
        """Initialize the agent."""
        # Set multiprocessing start method in case this is the main process
        if multiprocessing.current_process().name == 'MainProcess':
            try:
                multiprocessing.set_start_method('spawn', force=True)
                print("Set multiprocessing start method to 'spawn'")
            except RuntimeError:
                print("Could not set start method to 'spawn' (likely already set)")
                
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.clip_gate = ClipGate(device=str(self.device))
    
    def _extract_frames(self, video_path: pathlib.Path, fps: float = 1) -> List[np.ndarray]:
        """Extract frames from video at specified FPS."""
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / video_fps
        
        # Calculate frame sampling interval
        interval = int(video_fps / fps)
        interval = max(1, interval)
        
        # Extract frames
        frames = []
        frame_indices = []
        for i in range(0, frame_count, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                frame_indices.append(i)
                
        cap.release()
        
        print(f"📊 Extracted {len(frames)} frames at {fps} FPS from video (duration: {duration:.1f}s, {frame_count} total frames)")
        return frames, frame_indices
    
    async def _call_asr_service(self, audio_path: pathlib.Path, segments: List[Tuple[float, float]] = None) -> str:
        """Call ASR service to transcribe audio."""
        # Always use our fixed local implementation
        return whisper_transcribe_segments(audio_path)
    
    def _extract_entities_regex(self, text: str) -> List[str]:
        """Extract potential named entities using regex patterns."""
        entities = set()
        
        # Extract proper nouns (capitalized words not at start of sentence)
        proper_nouns = re.findall(r'(?<!^)(?<!\. )(?<!\? )(?<!\! )[A-Z][a-z]+', text)
        entities.update(proper_nouns)
        
        # Extract quoted phrases
        quotes = re.findall(r'"([^"]+)"', text)
        entities.update(quotes)
        
        # Extract hashtags
        hashtags = re.findall(r'#(\w+)', text)
        entities.update(hashtags)
        
        # Extract specific patterns like restaurant/place names, locations
        places = re.findall(r'(?:at|in|to|from|visit|go to) ([A-Z][a-zA-Z\s\']+)(?:\.|\,|\s|$)', text)
        entities.update(places)
        
        # Filter out common words and short entities
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "but", "or", "not", "for", "this", "that"}
        filtered_entities = [e.strip() for e in entities if len(e) > 3 and e.lower() not in stop_words]
        
        # Limit to reasonable number
        return list(set(filtered_entities))[:10]
    
    def _encode_image_for_gpt(self, frame: np.ndarray) -> str:
        """Convert an OpenCV frame to a base64 encoded string for GPT Vision."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and save to buffer
        pil_image = Image.fromarray(rgb_frame)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        
        # Encode to base64
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    async def _analyze_with_gpt_vision(self, frames: List[np.ndarray], prompt: str = None) -> Dict[str, Any]:
        """
        Analyze frames using GPT Vision API.
        
        Args:
            frames: List of frames to analyze
            prompt: Optional custom prompt for GPT Vision
            
        Returns:
            Dictionary with analysis results
        """
        if not frames:
            return {"error": "No frames provided for analysis"}
        
        start_time = time.time()
        
        # Select key frames to analyze (max 3 to keep costs down)
        if len(frames) > 3:
            # Take first, middle, and last frames
            selected_frames = [
                frames[0],
                frames[len(frames) // 2],
                frames[-1]
            ]
        else:
            selected_frames = frames
        
        # Default prompt for scene analysis
        if not prompt:
            prompt = """
            Analyze these frames from a video and provide the following information in JSON format:
            1. What type of place is shown (e.g., restaurant, cafe, hotel, landmark, etc.)?
            2. What objects are visible in the scene?
            3. What is the overall scene or setting (e.g., indoor dining, beach, city street)?
            4. Any specific activities happening?
            
            Format your response as a JSON object with these keys:
            {
                "place_type": "Type of place/establishment",
                "objects": ["list", "of", "objects"],
                "scene": "Description of scene/setting",
                "activities": ["list", "of", "activities"]
            }
            
            If this appears to be food-related content, also include:
            {
                "food_items": [
                    {"name": "dish name", "description": "brief description"}
                ]
            }
            """
        
        # Encode frames to base64
        encoded_frames = [self._encode_image_for_gpt(frame) for frame in selected_frames]
        
        try:
            # Prepare messages for GPT-4 Vision
            messages = [
                {
                    "role": "system", 
                    "content": "You are an expert video analyst specializing in location and food identification."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in encoded_frames]
                    ]
                }
            ]
            
            # Call GPT-4 Vision API
            response = openai_client.chat.completions.create(
                model="gpt-4o",  # Updated model name
                messages=messages,
                max_tokens=1200,
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            try:
                analysis = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                # Fall back to extracting JSON from the text
                text_response = response.choices[0].message.content
                json_match = re.search(r'```json\s*(.*?)\s*```', text_response, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(1))
                else:
                    # Try one more approach - find JSON block
                    json_match = re.search(r'{.*}', text_response, re.DOTALL)
                    if json_match:
                        analysis = json.loads(json_match.group(0))
                    else:
                        analysis = {"error": "Could not parse JSON from GPT response"}
            
            # Add processing metadata
            analysis["processing_time"] = time.time() - start_time
            analysis["frames_analyzed"] = len(selected_frames)
            analysis["model"] = "gpt-4-vision-preview"
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error in GPT Vision analysis: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}
    
    async def _analyze_food_specific(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Perform food-specific analysis on frames.
        
        Args:
            frames: List of frames to analyze
            
        Returns:
            Dictionary with food analysis results
        """
        if not frames:
            return {"error": "No frames provided for food analysis"}
        
        start_time = time.time()
        
        # Custom prompt for food analysis
        food_prompt = """
        Analyze these food images in detail and provide the following information in JSON format:
        
        1. What specific dishes are shown?
        2. What cuisine type is this?
        3. What ingredients can you identify?
        4. What restaurant setting is shown (casual, upscale, fast food, etc.)?
        
        Format your response as a JSON object with these keys:
        {
            "dishes": [
                {"name": "dish name", "description": "detailed description", "estimated_price_range": "$ or $$ or $$$"}
            ],
            "cuisine_type": "type of cuisine",
            "ingredients": ["list", "of", "visible", "ingredients"],
            "restaurant_setting": "description of setting",
            "meal_type": "breakfast/lunch/dinner/dessert"
        }
        
        Be specific and detailed in your analysis of the food items.
        """
        
        # Call GPT Vision with food-specific prompt
        return await self._analyze_with_gpt_vision(frames, food_prompt)
    
    
    
    async def stage0_metadata(self, state: ExtractionState) -> ExtractionState:
            """Stage 0: Extract metadata (caption, etc.)."""
            stage_start = time.time()
            print("🚀 Starting Stage 0: Metadata extraction")
            
            try:
                # Fetch caption/description
                caption_start = time.time()
                state.caption_text = fetch_caption(state.url)
                state.performance_profile["caption_time"] = time.time() - caption_start
                print(f"📝 Caption: {state.caption_text[:100]}..." if len(state.caption_text) > 100 else state.caption_text)
                
                # Extract initial entities from caption
                if state.caption_text:
                    state.entities = self._extract_entities_regex(state.caption_text)
                    print(f"🔍 Extracted entities: {', '.join(state.entities)}")
                
                # Mark stage as completed
                state.completed_stages.add(PipelineStage.STAGE_0_METADATA)
                state.current_stage = PipelineStage.STAGE_1_BASIC
                state.performance_profile["stage0_time"] = time.time() - stage_start
                
                return state
            except Exception as e:
                print(f"❌ Error in Stage 0: {e}")
                state.error = f"Stage 0 error: {str(e)}"
                return state
        
    async def stage1_basic(self, state: ExtractionState) -> ExtractionState:
        """Stage 1: VAD-sharded Whisper + regex NER + CLIP gate."""
        stage_start = time.time()
        print("🚀 Starting Stage 1: Basic extraction")
        
        try:
            # Extract audio
            audio_path = extract_audio(state.clip_path)
            
            # Use standard whisper transcription - avoid parallel ASR which may be causing issues
            asr_start = time.time()
            print("🎤 Transcribing audio with standard whisper (non-parallel)...")
            
            # Use simpler adaptive transcribe function directly
            state.speech_text = adaptive_whisper_transcribe(state.clip_path)
            state.performance_profile["asr_time"] = time.time() - asr_start
            
            print(f"📝 Transcription result: {state.speech_text[:100]}..." if len(state.speech_text) > 100 else state.speech_text)
            
            # Extract entities from speech
            speech_entities = self._extract_entities_regex(state.speech_text)
            state.entities.extend([e for e in speech_entities if e not in state.entities])
            if len(state.entities) > 10:  # Limit to top 10
                state.entities = state.entities[:10]
            print(f"🔍 Updated entities: {', '.join(state.entities)}")
            
            # Extract keyframes (1 FPS)
            frames_start = time.time()
            state.frames, frame_indices = self._extract_frames(state.clip_path, fps=1)
            state.performance_profile["frame_extraction_time"] = time.time() - frames_start
            
            # Run CLIP gate to filter frames based on entities
            if state.entities and state.frames:
                gate_start = time.time()
                gate_result = self.clip_gate.filter_frames(
                    state.frames, 
                    state.entities, 
                    threshold=0.2,
                    max_frames=5  # Reduce to 5 frames for GPT Vision
                )
                state.performance_profile["clip_gate_time"] = time.time() - gate_start
                
                # Get gated frames
                state.gated_frame_indices = gate_result["kept_frame_indices"]
                state.gated_frames = [state.frames[i] for i in state.gated_frame_indices]
                
                print(f"🔍 CLIP gate selected {len(state.gated_frames)}/{len(state.frames)} frames")
            else:
                # If no entities or frames, use all frames (up to 5)
                state.gated_frames = state.frames[:5]
                state.gated_frame_indices = list(range(min(5, len(state.frames))))
                print(f"⚠️ No CLIP gating applied. Using first {len(state.gated_frames)} frames.")
            
            # Mark stage as completed
            state.completed_stages.add(PipelineStage.STAGE_1_BASIC)
            state.current_stage = PipelineStage.STAGE_2_VISION
            state.performance_profile["stage1_time"] = time.time() - stage_start
            
            return state
        except Exception as e:
            print(f"❌ Error in Stage 1: {e}")
            import traceback
            traceback.print_exc()
            state.error = f"Stage 1 error: {str(e)}"
            return state
    async def stage2_vision(self, state: ExtractionState) -> ExtractionState:
        """Stage 2: GPT-Vision analysis on gated frames + food analysis."""
        stage_start = time.time()
        print("🚀 Starting Stage 2: Vision processing")
        
        try:
            # Only proceed if we have gated frames
            if state.gated_frames:
                # Run OCR on gated frames
                ocr_start = time.time()
                
                # Use parallel OCR directly on gated frames
                from parallel_ocr import process_frame_batch
                
                # Add regions to frames for OCR processing
                frames_with_regions = [(0, frame, []) for frame in state.gated_frames]
                
                # Process in a single batch since we already filtered frames
                ocr_results = process_frame_batch(frames_with_regions)
                
                # Combine OCR text from all frames
                all_texts = [text for _, text in ocr_results if text]
                state.frame_text = "\n".join(all_texts)
                
                state.performance_profile["ocr_time"] = time.time() - ocr_start
                print(f"📝 OCR text: {state.frame_text[:100]}..." if len(state.frame_text) > 100 else state.frame_text)
                
                # Check if we already have parsed_info from Stage 1 with multiple activities
                # This would happen if caption/speech text already revealed multiple places
                if state.parsed_info and "activities" in state.parsed_info and len(state.parsed_info["activities"]) > 1:
                    print(f"🔍 Detected compilation with {len(state.parsed_info['activities'])} activities")
                    
                    # Store vision results for each activity
                    state.visual_results = []
                    
                    # Process each activity separately
                    for i, activity in enumerate(state.parsed_info["activities"]):
                        # For each activity, ideally we would have specific frames
                        # If we don't have a way to associate frames with activities yet, 
                        # we can divide frames evenly or use all frames for each
                        activity_frames = state.gated_frames  # For now, use all frames for each activity
                        
                        # Get place-specific entities for CLIP filtering
                        place_entities = [activity.get("place_name", "")]
                        if "genre" in activity and activity["genre"]:
                            place_entities.append(activity["genre"])
                        if "cuisine" in activity and activity["cuisine"]:
                            place_entities.append(activity["cuisine"])
                        
                        # Filter frames specific to this activity using CLIP if we have entities
                        if place_entities and len(state.gated_frames) > 5:
                            try:
                                gate_result = self.clip_gate.filter_frames(
                                    state.gated_frames, 
                                    place_entities, 
                                    threshold=0.15,  # Lower threshold for better recall
                                    max_frames=3  # Fewer frames per activity
                                )
                                activity_frames = [state.gated_frames[i] for i in gate_result["kept_frame_indices"]]
                                print(f"🔍 Activity {i+1}: CLIP gate selected {len(activity_frames)}/{len(state.gated_frames)} frames")
                            except Exception as e:
                                print(f"⚠️ Error in CLIP gating for activity {i+1}: {e}")
                        
                        # If no frames were selected or CLIP failed, use a subset of frames
                        if not activity_frames:
                            # Use frames based on activity index in a compilation
                            frame_count = len(state.gated_frames)
                            start_idx = (i * frame_count) // len(state.parsed_info["activities"])
                            end_idx = ((i + 1) * frame_count) // len(state.parsed_info["activities"])
                            activity_frames = state.gated_frames[start_idx:end_idx]
                            if not activity_frames:
                                activity_frames = [state.gated_frames[0]]  # Fallback to first frame
                        
                        # Create custom prompt for this specific activity
                        custom_prompt = f"""
                        Analyze these frames for a video showing {activity.get('place_name', 'a place')} 
                        which is a {activity.get('genre', 'place')}
                        {f"with {activity.get('cuisine', '')} cuisine" if activity.get('cuisine') else ''}.
                        
                        Provide the following information in JSON format:
                        1. What specific objects related to this {activity.get('genre', 'place')} are visible?
                        2. What is the overall scene or setting?
                        3. Any specific activities happening?
                        
                        Format as a JSON with these keys:
                        {{
                            "objects": ["list", "of", "objects"],
                            "scene": "Description of scene/setting",
                            "activities": ["list", "of", "activities"]
                        }}
                        
                        {"Additionally, analyze the food items visible:" if activity.get('genre') in ['restaurant', 'cafe', 'bakery', 'food'] else ""}
                        {"{ 'food_items': [ {'name': 'dish name', 'description': 'brief description'} ] }" if activity.get('genre') in ['restaurant', 'cafe', 'bakery', 'food'] else ""}
                        """
                        
                        # Analyze frames with GPT Vision for this activity
                        vision_analysis = await self._analyze_with_gpt_vision(activity_frames, custom_prompt)
                        
                        # Add the result to visual_results
                        state.visual_results.append({
                            "model_type": "gpt_vision",
                            "activity_index": i,
                            "place_name": activity.get("place_name", ""),
                            "confidence": 0.9,
                            "analysis": vision_analysis,
                            "processing_time": vision_analysis.get("processing_time", 0),
                            "frames_processed": len(activity_frames)
                        })
                        
                        # If food-related, do additional food-specific analysis
                        if activity.get('genre') in ['restaurant', 'cafe', 'bakery', 'food']:
                            food_analysis = await self._analyze_food_specific(activity_frames)
                            
                            # Add the food result
                            state.visual_results.append({
                                "model_type": "food_analysis",
                                "activity_index": i,
                                "place_name": activity.get("place_name", ""),
                                "confidence": 0.9,
                                "analysis": food_analysis,
                                "processing_time": food_analysis.get("processing_time", 0),
                                "frames_processed": len(activity_frames)
                            })
                else:
                    # Single activity case - run standard analysis
                    vision_start = time.time()
                    vision_analysis = await self._analyze_with_gpt_vision(state.gated_frames)
                    
                    # Check if this is food-related content
                    food_related = False
                    if vision_analysis.get("place_type", "").lower() in ["restaurant", "cafe", "bakery", "food truck", "dining"]:
                        food_related = True
                    elif "food_items" in vision_analysis and vision_analysis["food_items"]:
                        food_related = True
                    
                    # If food-related, do additional food-specific analysis
                    food_analysis = None
                    if food_related:
                        print("🍔 Detected food-related content, running specialized food analysis")
                        food_analysis = await self._analyze_food_specific(state.gated_frames)
                        
                    # Store the results
                    visual_results = [{
                        "model_type": "gpt_vision",
                        "activity_index": 0,  # Single activity
                        "confidence": 0.9,
                        "analysis": vision_analysis,
                        "processing_time": vision_analysis.get("processing_time", 0),
                        "frames_processed": vision_analysis.get("frames_analyzed", 0)
                    }]
                    
                    if food_analysis:
                        visual_results.append({
                            "model_type": "food_analysis",
                            "activity_index": 0,  # Single activity
                            "confidence": 0.9,
                            "analysis": food_analysis,
                            "processing_time": food_analysis.get("processing_time", 0),
                            "frames_processed": food_analysis.get("frames_analyzed", 0)
                        })
                    
                    state.visual_results = visual_results
                    state.performance_profile["vision_time"] = time.time() - vision_start
                
                # Log vision results
                for result in state.visual_results:
                    model_type = result.get("model_type", "unknown")
                    activity_index = result.get("activity_index", 0)
                    analysis = result.get("analysis", {})
                    if model_type == "gpt_vision":
                        place_type = analysis.get("place_type", "unknown")
                        objects = analysis.get("objects", [])[:5]
                        print(f"🔍 Activity {activity_index+1}: GPT Vision detected place type: {place_type}")
                        if objects:
                            print(f"🔍 Activity {activity_index+1}: GPT Vision detected objects: {', '.join(objects[:5])}")
                    elif model_type == "food_analysis":
                        dishes = [dish.get("name", "unknown") for dish in analysis.get("dishes", [])]
                        if dishes:
                            print(f"🍔 Activity {activity_index+1}: Food analysis detected dishes: {', '.join(dishes)}")
            
            # Mark stage as completed
            state.completed_stages.add(PipelineStage.STAGE_2_VISION)
            state.current_stage = PipelineStage.STAGE_3_FUSION
            state.performance_profile["stage2_time"] = time.time() - stage_start
            
            return state
        except Exception as e:
            print(f"❌ Error in Stage 2: {e}")
            state.error = f"Stage 2 error: {str(e)}"
            return state
    
    async def stage3_fusion(self, state: ExtractionState) -> ExtractionState:
        """Stage 3: Single GPT-4o fusion → DB → UI."""
        stage_start = time.time()
        print("🚀 Starting Stage 3: LLM fusion")
        
        try:
            # Combine all text sources
            fused_text = "\n".join(
                filter(
                    None,
                    [
                        f"SPEECH: {state.speech_text}" if state.speech_text else "",
                        f"OCR TEXT: {state.frame_text}" if state.frame_text else "",
                        f"CAPTION: {state.caption_text}" if state.caption_text else "",
                    ],
                )
            )
            
            # Add visual information from GPT Vision
            visual_text = []
            for result in state.visual_results:
                model_type = result.get("model_type", "unknown")
                analysis = result.get("analysis", {})
                activity_idx = result.get("activity_index", 0)
                activity_prefix = f"ACTIVITY {activity_idx+1}: " if "activity_index" in result else ""
                
                if model_type == "gpt_vision":
                    # Add place type
                    if "place_type" in analysis:
                        visual_text.append(f"{activity_prefix}PLACE TYPE: {analysis['place_type']}")
                    
                    # Add objects
                    if "objects" in analysis and analysis["objects"]:
                        objects_text = ", ".join(analysis["objects"])
                        visual_text.append(f"{activity_prefix}DETECTED OBJECTS: {objects_text}")
                    
                    # Add scene
                    if "scene" in analysis:
                        visual_text.append(f"{activity_prefix}SCENE: {analysis['scene']}")
                    
                    # Add activities
                    if "activities" in analysis and analysis["activities"]:
                        activities_text = ", ".join(analysis["activities"])
                        visual_text.append(f"{activity_prefix}ACTIVITIES: {activities_text}")
                    
                    # Add food items
                    if "food_items" in analysis and analysis["food_items"]:
                        food_items = [item.get("name", "") for item in analysis["food_items"]]
                        food_text = ", ".join(food_items)
                        visual_text.append(f"{activity_prefix}FOOD ITEMS: {food_text}")
                
                elif model_type == "food_analysis":
                    # Add cuisine type
                    if "cuisine_type" in analysis:
                        visual_text.append(f"{activity_prefix}CUISINE TYPE: {analysis['cuisine_type']}")
                    
                    # Add dishes
                    if "dishes" in analysis and analysis["dishes"]:
                        dishes = [f"{dish.get('name', '')} ({dish.get('estimated_price_range', '')})" 
                                for dish in analysis["dishes"]]
                        dishes_text = ", ".join(dishes)
                        visual_text.append(f"{activity_prefix}DISHES: {dishes_text}")
                    
                    # Add ingredients
                    if "ingredients" in analysis and analysis["ingredients"]:
                        ingredients_text = ", ".join(analysis["ingredients"])
                        visual_text.append(f"{activity_prefix}INGREDIENTS: {ingredients_text}")
                    
                    # Add restaurant setting
                    if "restaurant_setting" in analysis:
                        visual_text.append(f"{activity_prefix}RESTAURANT SETTING: {analysis['restaurant_setting']}")
                    
                    # Add meal type
                    if "meal_type" in analysis:
                        visual_text.append(f"{activity_prefix}MEAL TYPE: {analysis['meal_type']}")
            
            if visual_text:
                fused_text += "\n\nVISUAL ANALYSIS:\n" + "\n".join(visual_text)
            
            print(f"🔹 Text fused for LLM ({len(fused_text)} chars)")
            
            # Parse structured info from fused text
            parser_start = time.time()
            parsed_info = parse_place_info(fused_text)
            state.performance_profile["llm_parse_time"] = time.time() - parser_start
            
            # Store the parsed info directly (it's already a dictionary now)
            state.parsed_info = parsed_info
            
            # Run parallel geocoding for places
            geo_start = time.time()
            if "activities" in state.parsed_info:
                state.parsed_info["activities"] = parallel_geocode_activities(state.parsed_info["activities"])
            state.performance_profile["geocode_time"] = time.time() - geo_start
            
            # Enhance with visual data for each activity
            if state.visual_results and "activities" in state.parsed_info:
                # Group visual results by activity_index
                activity_visual_results = {}
                for result in state.visual_results:
                    activity_idx = result.get("activity_index", 0)
                    if activity_idx not in activity_visual_results:
                        activity_visual_results[activity_idx] = []
                    activity_visual_results[activity_idx].append(result)
                
                # Process each activity
                for idx, activity in enumerate(state.parsed_info["activities"]):
                    # Get visual results for this activity
                    activity_results = activity_visual_results.get(idx, [])
                    
                    # Initialize visual_data if it doesn't exist
                    if "visual_data" not in activity:
                        activity["visual_data"] = {
                            "detected_objects": [],
                            "scene_categories": [],
                            "food_items": []
                        }
                    
                    # Add visual data from each result for this activity
                    for result in activity_results:
                        model_type = result.get("model_type", "")
                        analysis = result.get("analysis", {})
                        
                        if model_type == "gpt_vision":
                            # Add objects
                            for obj in analysis.get("objects", []):
                                activity["visual_data"]["detected_objects"].append({
                                    "label": obj,
                                    "confidence": 0.9
                                })
                            
                            # Add scene categories
                            if "scene" in analysis:
                                activity["visual_data"]["scene_categories"].append({
                                    "category": analysis["scene"],
                                    "confidence": 0.9
                                })
                            
                            # Add place type if available and place_name isn't set
                            if "place_type" in analysis and not activity.get("place_name"):
                                activity["place_name"] = analysis["place_type"]
                            
                            # Add food items
                            for food in analysis.get("food_items", []):
                                activity["visual_data"]["food_items"].append({
                                    "name": food.get("name", "Unknown food"),
                                    "confidence": 0.9,
                                    "description": food.get("description", "")
                                })
                        
                        elif model_type == "food_analysis":
                            analysis = result.get("analysis", {})
                            
                            # Add cuisine type to genre if not already specified
                            if "cuisine_type" in analysis and not activity.get("cuisine"):
                                activity["cuisine"] = analysis["cuisine_type"]
                            
                            # Add dishes to food items
                            for dish in analysis.get("dishes", []):
                                found = False
                                for existing in activity["visual_data"]["food_items"]:
                                    if dish.get("name", "").lower() in existing["name"].lower():
                                        # Update existing entry
                                        existing["description"] = dish.get("description", "")
                                        existing["price_range"] = dish.get("estimated_price_range", "")
                                        found = True
                                        break
                                
                                if not found:
                                    activity["visual_data"]["food_items"].append({
                                        "name": dish.get("name", "Unknown dish"),
                                        "confidence": 0.9,
                                        "description": dish.get("description", ""),
                                        "price_range": dish.get("estimated_price_range", "")
                                    })
                            
                            # Add restaurant setting to vibes if not already specified
                            if "restaurant_setting" in analysis and not activity.get("vibes"):
                                activity["vibes"] = analysis["restaurant_setting"]
                
                # If any activities still lack visual data, assign results from most similar activity
                activities_with_visual = [idx for idx, activity in enumerate(state.parsed_info["activities"]) 
                                        if "visual_data" in activity and activity["visual_data"]["detected_objects"]]
                
                for idx, activity in enumerate(state.parsed_info["activities"]):
                    if idx not in activities_with_visual and activities_with_visual:
                        # Find most similar activity that has visual data
                        # This is a simple approach - you could use more sophisticated similarity measures
                        similar_idx = activities_with_visual[0]
                        activity["visual_data"] = state.parsed_info["activities"][similar_idx]["visual_data"]
                        print(f"⚠️ Activity {idx+1} ({activity.get('place_name', '')}) lacks visual data, borrowing from activity {similar_idx+1}")
            
            # Add recognition data to the result
            state.parsed_info["recognition_data"] = {
                "recognition_results": state.visual_results,
                "processing_time": state.performance_profile.get("vision_time", 0),
                "frames_processed": len(state.gated_frames),
                "models_used": ["gpt_vision", "food_analysis"] if any(r["model_type"] == "food_analysis" for r in state.visual_results) else ["gpt_vision"]
            }
            
            # Add raw extraction texts
            state.parsed_info["speech_text"] = state.speech_text
            state.parsed_info["frame_text"] = state.frame_text
            state.parsed_info["caption_text"] = state.caption_text
            
            # Add performance profile
            state.parsed_info["performance_profile"] = state.performance_profile
            
            # Mark stage as completed
            state.completed_stages.add(PipelineStage.STAGE_3_FUSION)
            state.performance_profile["stage3_time"] = time.time() - stage_start
            state.performance_profile["total_time"] = time.time() - state.started_at
            
            return state
        except Exception as e:
            print(f"❌ Error in Stage 3: {e}")
            import traceback
            traceback.print_exc()
            state.error = f"Stage 3 error: {str(e)}"
            return state


# For compatibility with existing code
def parallel_geocode_activities(activities):
    """Process geocoding for multiple activities in parallel."""
    if not activities:
        return activities

    # Prepare geocoding tasks
    geocode_tasks = []

    for i, activity in enumerate(activities):
        if not activity.get('place_name'):
            continue

        # Build hint per activity
        hints = []
        availability = activity.get('availability', {})
        if availability.get('city'):
            hints.append(availability.get('city'))
        if availability.get('state'):
            hints.append(availability.get('state'))
        if availability.get('country'):
            hints.append(availability.get('country'))
        extra_hint = ", ".join(hints) if hints else None

        # Skip if already has street address
        if not availability.get('street_address'):
            geocode_tasks.append(
                {
                    "index": i,
                    "place_name": activity.get('place_name'),
                    "genre": activity.get('genre'),
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
                        activities[activity_index]['availability']['street_address'] = geo.get("display_address")

                        # Only update these fields if they're empty
                        if not activities[activity_index]['availability'].get('city'):
                            activities[activity_index]['availability']['city'] = geo.get("city")
                        if not activities[activity_index]['availability'].get('state'):
                            activities[activity_index]['availability']['state'] = geo.get("state")
                        if not activities[activity_index]['availability'].get('country'):
                            activities[activity_index]['availability']['country'] = geo.get("country")
                        if not activities[activity_index]['availability'].get('region'):
                            activities[activity_index]['availability']['region'] = geo.get("region")
                    else:
                        print(f"❌ Geocode failed for: {task['place_name']}")
                except Exception as e:
                    print(f"❌ Geocode error for {task['place_name']}: {str(e)}")

    return activities


# Main execution function (compatible with old API)
async def run_async(url: str, enable_visual_recognition: bool = False):
    """
    Process a single short‑form video URL and return parsed info with waterfall processing.
    
    Args:
        url: URL of the video to process
        enable_visual_recognition: Whether to enable visual recognition (default: False)
    
    Returns:
        Dictionary with extraction results
    """
    agent = Agent(use_gpu=True)
    result = await agent.process_video(url)
    return result


# Synchronous wrapper for compatibility
def run(url: str, enable_visual_recognition: bool = False):
    """
    Synchronous wrapper around run_async for backward compatibility.
    """
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(run_async(url, enable_visual_recognition))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Reel / TikTok / Short URL")
    ap.add_argument("--out", default="result.json", help="Output JSON file path")
    ap.add_argument(
        "--time", action="store_true", help="Show detailed timing information"
    )
    ap.add_argument(
        "--enable-visual", action="store_true", help="Enable visual recognition"
    )
    args = ap.parse_args()

    try:
        start_time = time.time()
        print(f"🎬 Processing video from: {args.url}")
        
        # Run asynchronously
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(run_async(args.url, args.enable_visual))

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
                
                # Print visual data if available
                if "visual_data" in activity:
                    # Print top detected objects
                    objects = activity["visual_data"].get("detected_objects", [])
                    if objects:
                        print(f"    📦 Top objects: {', '.join([obj['label'] for obj in objects[:3]])}")
                    
                    # Print top scenes
                    scenes = activity["visual_data"].get("scene_categories", [])
                    if scenes:
                        print(f"    🏞️ Top scenes: {', '.join([scene['category'] for scene in scenes[:3]])}")
                    
                    # Print food items
                    food_items = activity["visual_data"].get("food_items", [])
                    if food_items:
                        print(f"    🍔 Food items: {', '.join([food['name'] for food in food_items[:3]])}")

            # Print performance profile if requested
            if args.time and "performance_profile" in data:
                profile = data["performance_profile"]
                print("\n⏱️ Performance Profile:")
                print(f"  📥 Download: {profile.get('download_time', 0):.2f}s")
                print(f"  📝 Stage 0 (Metadata): {profile.get('stage0_time', 0):.2f}s")
                print(f"  🔍 Stage 1 (Basic): {profile.get('stage1_time', 0):.2f}s")
                print(f"  🎬 Stage 2 (Vision): {profile.get('stage2_time', 0):.2f}s")
                print(f"  🧠 Stage 3 (Fusion): {profile.get('stage3_time', 0):.2f}s")
                print(f"  🏁 Total Time: {profile.get('total_time', 0):.2f}s")

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