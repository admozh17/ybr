#!/usr/bin/env python3
"""
GPT-Centered Video Recognition Module
This module uses GPT-4V as the primary analysis method for video content.
"""

import os
import cv2
import time
import numpy as np
import pathlib
import json
from io import BytesIO
from enum import Enum
import concurrent.futures
from typing import Dict, List, Any, Tuple, Optional
import traceback

# Check if OpenAI is available
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️ OpenAI package not found. Install with: pip install openai")

# Analysis modes
class AnalysisMode(Enum):
    """Video analysis modes"""
    BASIC = "basic"       # Basic analysis only
    STANDARD = "standard" # Standard GPT-4V analysis
    DETAILED = "detailed" # Detailed GPT-4V analysis with more frames

class GPTVideoRecognition:
    """
    Video recognition system that uses GPT-4V as the primary analysis method.
    """
    
    def __init__(self, 
                 max_frames: int = 5, 
                 analysis_mode: AnalysisMode = AnalysisMode.STANDARD):
        """
        Initialize the GPT video recognition system.
        
        Args:
            max_frames: Maximum number of frames to process per video
            analysis_mode: Which analysis mode to use
        """
        self.max_frames = max_frames
        self.analysis_mode = analysis_mode
        
        # Adjust max frames based on mode
        if analysis_mode == AnalysisMode.BASIC:
            self.max_frames = min(3, max_frames)  # Use fewer frames for basic mode
        elif analysis_mode == AnalysisMode.DETAILED:
            self.max_frames = max(8, max_frames)  # Use more frames for detailed mode
        
        # Check OpenAI API
        self.has_openai = HAS_OPENAI and (os.getenv("OPENAI_API_KEY") is not None)
        if not self.has_openai:
            print("❌ OpenAI API not available. Please set OPENAI_API_KEY environment variable.")
            print("   Video recognition will not work without OpenAI API access.")
        
        print(f"🎥 Initialized GPTVideoRecognition")
        print(f"   Analysis mode: {self.analysis_mode.value}")
        print(f"   Max frames: {self.max_frames}")
        print(f"   GPT-4V available: {self.has_openai}")

    def extract_frames(self, video_path: pathlib.Path, num_frames: int = None) -> List[np.ndarray]:
        """
        Extract frames from video for processing.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (None for default)
            
        Returns:
            List of extracted frames as numpy arrays
        """
        if num_frames is None:
            num_frames = self.max_frames
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps
        
        # Extract key frames - beginning, middle, end and evenly spaced
        frames = []
        
        # If few frames, just use them all
        if frame_count <= num_frames:
            for i in range(frame_count):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
        else:
            # Always include first and last frame
            key_positions = [0, frame_count-1]
            
            # Add evenly spaced frames in between
            if num_frames > 2:
                additional_frames = num_frames - 2
                for i in range(1, additional_frames+1):
                    # Calculate position for evenly spaced frames
                    pos = int(i * frame_count / (additional_frames+1))
                    key_positions.append(pos)
            
            # Sort positions and remove duplicates
            key_positions = sorted(set(key_positions))
            
            # Extract frames
            for pos in key_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                
        cap.release()
        
        print(f"📊 Extracted {len(frames)} frames from video (duration: {duration:.1f}s, {frame_count} total frames)")
        return frames

    async def analyze_with_gpt4v(self, 
                              frames: List[np.ndarray],
                              content_type: str = None,
                              speech_text: str = "",
                              ocr_text: str = "",
                              caption_text: str = "") -> Dict[str, Any]:
        """
        Analyze video frames using GPT-4V.
        
        Args:
            frames: List of video frames
            content_type: Content type hint
            speech_text: Transcribed speech from video
            ocr_text: OCR text from video frames
            caption_text: Video caption text
            
        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        
        if not self.has_openai:
            return {
                "error": "OpenAI API not available",
                "processing_time": 0
            }
        
        try:
            # Encode frames as base64
            import base64
            from PIL import Image
            
            encoded_images = []
            for frame in frames:
                # Convert from BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Resize image to reduce token usage if needed
                if pil_image.width > 1024 or pil_image.height > 1024:
                    pil_image.thumbnail((1024, 1024), Image.LANCZOS)
                
                buffer = BytesIO()
                pil_image.save(buffer, format="JPEG", quality=80)
                encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                encoded_images.append(encoded_image)
            
            # Build context from available text
            context = ""
            if speech_text:
                context += f"Speech transcription: {speech_text}\n\n"
            if ocr_text:
                context += f"Text visible in the video: {ocr_text}\n\n"
            if caption_text:
                context += f"Video caption: {caption_text}\n\n"
                
            if content_type:
                context += f"Content type: {content_type}\n\n"
            
            # Prepare prompt based on analysis mode
            if self.analysis_mode == AnalysisMode.BASIC:
                prompt = f"""
                Please analyze these frames from a short-form video (like Instagram Reel, TikTok, or YouTube Short).
                
                {context}
                
                Provide a brief analysis focusing on:
                1. What type of place is shown (restaurant, hotel, outdoors, etc.)
                2. What food items are visible (if any)
                3. The main activities happening in the video
                
                Format your response as a JSON object with:
                {{
                    "scene_description": "Brief description of what's happening",
                    "location_type": "Restaurant/Home/Outdoors/etc",
                    "food_items": ["List", "of", "visible", "food", "items"],
                    "activities": ["List", "of", "activities"],
                    "confidence": 0.XX
                }}
                """
            else:
                # Standard or detailed mode
                prompt = f"""
                Please analyze these frames from a short-form video (like Instagram Reel, TikTok, or YouTube Short).
                
                {context}
                
                I need a detailed description of what's happening in this video. Please tell me:
                
                1. What activities are people doing in this video?
                2. Is this at a restaurant, home, or other location? Be specific about the place.
                3. What food items are visible and what type of cuisine is shown?
                4. What's the overall context and purpose of this video?
                5. Is this a product review, personal experience, travel vlog, or something else?
                6. What specific details can you see about the environment or setting?
                
                Format your response as a JSON object with the following structure:
                {{
                    "scene_description": "Detailed description of what's happening in the video",
                    "location_type": "Restaurant/Home/Outdoors/etc",
                    "location_details": "Specific details about the location if identifiable",
                    "cuisine_type": "Type of cuisine if food is shown",
                    "food_items": ["List", "of", "visible", "food", "items"],
                    "food_details": "Details about the food preparation, presentation, etc.",
                    "activities": ["List", "of", "activities", "happening"],
                    "activity_details": "Detailed description of the main activities",
                    "video_purpose": "Review/Personal/Tutorial/etc",
                    "environment": "Details about the environment/setting",
                    "notable_objects": ["List", "of", "important", "objects"],
                    "confidence": 0.XX (how confident you are in this analysis, from 0-1)
                }}
                """
            
            # Call GPT-4V API
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in encoded_images]
                        ]
                    }
                ],
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            response_json = json.loads(response_text)
            
            # Add metadata
            result = {
                "gpt4v_analysis": response_json,
                "processing_time": time.time() - start_time,
                "frames_analyzed": len(frames),
                "analysis_mode": self.analysis_mode.value
            }
            
            # Generate activity summary
            result["activity_summary"] = self.generate_activity_summary(response_json)
            
            return result
            
        except Exception as e:
            print(f"❌ Error in GPT-4V analysis: {e}")
            traceback.print_exc()
            return {
                "error": f"GPT-4V analysis failed: {str(e)}",
                "processing_time": time.time() - start_time
            }
            
    def generate_activity_summary(self, gpt4v_result: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary from GPT-4V analysis.
        
        Args:
            gpt4v_result: GPT-4V analysis results
            
        Returns:
            Human-readable summary
        """
        summary_parts = []
        
        # Title
        summary_parts.append("# Video Content Analysis")
        
        # Main description
        if "scene_description" in gpt4v_result:
            summary_parts.append("\n## Overview")
            summary_parts.append(gpt4v_result["scene_description"])
        
        # Location information
        location_parts = []
        if "location_type" in gpt4v_result:
            location_parts.append(f"**Type:** {gpt4v_result['location_type']}")
        if "location_details" in gpt4v_result and gpt4v_result["location_details"]:
            location_parts.append(f"**Details:** {gpt4v_result['location_details']}")
        if "environment" in gpt4v_result and gpt4v_result["environment"]:
            location_parts.append(f"**Environment:** {gpt4v_result['environment']}")
            
        if location_parts:
            summary_parts.append("\n## Location")
            summary_parts.extend(location_parts)
        
        # Food information
        food_parts = []
        if "cuisine_type" in gpt4v_result and gpt4v_result["cuisine_type"]:
            food_parts.append(f"**Cuisine:** {gpt4v_result['cuisine_type']}")
        
        if "food_items" in gpt4v_result and gpt4v_result["food_items"]:
            food_parts.append("**Food Items:**")
            for item in gpt4v_result["food_items"]:
                food_parts.append(f"- {item}")
                
        if "food_details" in gpt4v_result and gpt4v_result["food_details"]:
            food_parts.append(f"**Details:** {gpt4v_result['food_details']}")
            
        if food_parts:
            summary_parts.append("\n## Food")
            summary_parts.extend(food_parts)
        
        # Activity information
        activity_parts = []
        if "activities" in gpt4v_result and gpt4v_result["activities"]:
            activity_parts.append("**Activities:**")
            for activity in gpt4v_result["activities"]:
                activity_parts.append(f"- {activity}")
                
        if "activity_details" in gpt4v_result and gpt4v_result["activity_details"]:
            activity_parts.append(f"**Details:** {gpt4v_result['activity_details']}")
            
        if "video_purpose" in gpt4v_result and gpt4v_result["video_purpose"]:
            activity_parts.append(f"**Video Purpose:** {gpt4v_result['video_purpose']}")
            
        if activity_parts:
            summary_parts.append("\n## Activities")
            summary_parts.extend(activity_parts)
        
        # Notable objects
        if "notable_objects" in gpt4v_result and gpt4v_result["notable_objects"]:
            summary_parts.append("\n## Notable Objects")
            for obj in gpt4v_result["notable_objects"]:
                summary_parts.append(f"- {obj}")
        
        # Confidence
        if "confidence" in gpt4v_result:
            confidence = float(gpt4v_result["confidence"])
            confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"
            summary_parts.append(f"\n**Analysis Confidence:** {confidence_level} ({confidence:.2f})")
        
        # Join parts
        return "\n".join(summary_parts)

    async def analyze_video(self, 
                         video_path: pathlib.Path,
                         content_type: Optional[str] = None,
                         speech_text: str = "",
                         ocr_text: str = "",
                         caption_text: str = "") -> Dict[str, Any]:
        """
        Main entry point for video analysis.
        
        Args:
            video_path: Path to video file
            content_type: Pre-classified content type if available
            speech_text: Transcribed speech from video
            ocr_text: OCR text from video frames
            caption_text: Video caption text
            
        Returns:
            Dictionary with recognition results
        """
        start_time = time.time()
        
        try:
            # Extract frames
            frames = self.extract_frames(video_path, self.max_frames)
            if not frames:
                return {"error": "Failed to extract frames from video"}
            
            # Run GPT-4V analysis
            result = await self.analyze_with_gpt4v(
                frames,
                content_type=content_type,
                speech_text=speech_text,
                ocr_text=ocr_text,
                caption_text=caption_text
            )
            
            # Add total processing time
            result["total_processing_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            print(f"❌ Error in video analysis: {e}")
            traceback.print_exc()
            return {
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def enhance_extraction_data(self, recognition_result: Dict[str, Any], extraction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance extraction data with recognition results.
        
        Args:
            recognition_result: Results from GPT video recognition
            extraction_data: Original extraction data
            
        Returns:
            Enhanced extraction data
        """
        try:
            # Make a copy to avoid modifying the original
            enhanced_data = extraction_data.copy()
            
            # Add recognition data
            enhanced_data["recognition_data"] = recognition_result
            
            # Add activity summary
            if "activity_summary" in recognition_result:
                enhanced_data["activity_summary"] = recognition_result["activity_summary"]
            
            # Enhance activities with visual data from GPT-4V
            if "activities" in enhanced_data and "gpt4v_analysis" in recognition_result:
                gpt_analysis = recognition_result["gpt4v_analysis"]
                
                for activity in enhanced_data["activities"]:
                    # Initialize visual_data if not present
                    if "visual_data" not in activity:
                        activity["visual_data"] = {
                            "detected_objects": [],
                            "scene_categories": [],
                            "food_items": [], 
                            "activity_description": ""
                        }
                    
                    # Add scene description
                    if "scene_description" in gpt_analysis:
                        activity["visual_data"]["activity_description"] = gpt_analysis["scene_description"]
                    
                    # Add location information
                    if "location_type" in gpt_analysis:
                        activity["visual_data"]["scene_categories"].append({
                            "category": gpt_analysis["location_type"],
                            "confidence": gpt_analysis.get("confidence", 0.8)
                        })
                    
                    # Add food items
                    if "food_items" in gpt_analysis:
                        for food in gpt_analysis["food_items"]:
                            activity["visual_data"]["food_items"].append({
                                "name": food,
                                "confidence": gpt_analysis.get("confidence", 0.8)
                            })
                    
                    # Add notable objects
                    if "notable_objects" in gpt_analysis:
                        for obj in gpt_analysis["notable_objects"]:
                            activity["visual_data"]["detected_objects"].append({
                                "label": obj,
                                "confidence": gpt_analysis.get("confidence", 0.8)
                            })
            
            return enhanced_data
        except Exception as e:
            print(f"❌ Error enhancing extraction data: {e}")
            traceback.print_exc()
            return extraction_data

# Synchronous wrapper for the asynchronous analyze_video method
def analyze_video_sync(recognition_system, video_path, **kwargs):
    """Synchronous wrapper for asynchronous analyze_video"""
    import asyncio
    return asyncio.run(recognition_system.analyze_video(video_path, **kwargs))

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT-Centered Video Recognition")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", default="gpt_video_analysis.json", help="Output JSON file")
    parser.add_argument("--mode", choices=["basic", "standard", "detailed"], 
                       default="standard", help="Analysis mode")
    parser.add_argument("--frames", type=int, default=5, help="Maximum frames to analyze")
    
    args = parser.parse_args()
    
    # Map string mode to enum
    mode_map = {
        "basic": AnalysisMode.BASIC,
        "standard": AnalysisMode.STANDARD,
        "detailed": AnalysisMode.DETAILED
    }
    
    # Initialize recognition system
    recognition = GPTVideoRecognition(
        max_frames=args.frames,
        analysis_mode=mode_map[args.mode]
    )
    
    # Process video
    video_path = pathlib.Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        exit(1)
    
    # Run analysis
    result = analyze_video_sync(
        recognition, 
        video_path
    )
    
    # Save result
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"✅ Analysis results saved to: {args.output}")
    
    # Print summary
    if "activity_summary" in result:
        print("\n" + "="*50)
        print(result["activity_summary"])
        print("="*50)
