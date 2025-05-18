#!/usr/bin/env python3
"""
Model selector for video recognition based on content type.
This module intelligently selects which recognition models to use
based on the content of the video.
"""

import os
import torch
import pathlib
from enum import Enum
from typing import List, Dict, Any, Optional

# Import from video_recognition module
from video_recognition import ModelType


class ContentCategory(Enum):
    """Categories of video content."""
    FOOD = "food"
    TRAVEL = "travel"
    SHOPPING = "shopping"
    FITNESS = "fitness"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    UNKNOWN = "unknown"


class ModelSelector:
    """
    Selects the appropriate model(s) for video recognition
    based on content category and available resources.
    """
    
    def __init__(self, device: str = None, 
                 resource_limit: str = "auto", 
                 api_enabled: bool = False):
        """
        Initialize the model selector.
        
        Args:
            device: Computing device ('cpu', 'cuda', or None for auto-detection)
            resource_limit: Resource usage limit ('low', 'medium', 'high', or 'auto')
            api_enabled: Whether to allow API-based models (GPT-4V, Claude Vision)
        """
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Resource limitation
        self.resource_limit = resource_limit
        if resource_limit == "auto":
            # Auto-detect based on available resources
            if self.device == "cpu":
                # CPU only - use lightweight models
                self.resource_limit = "low"
            elif torch.cuda.is_available():
                # Check GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory < 4 * (1024 ** 3):  # Less than 4GB
                    self.resource_limit = "low"
                elif gpu_memory < 8 * (1024 ** 3):  # Less than 8GB
                    self.resource_limit = "medium"
                else:  # 8GB or more
                    self.resource_limit = "high"
        
        # API usage permission
        self.api_enabled = api_enabled
        
        # Define model categories by resource usage
        self.model_resource_usage = {
            ModelType.YOLO_TINY: "low",
            ModelType.SCENE_CLASSIFICATION: "low",
            ModelType.YOLO_MEDIUM: "medium",
            ModelType.CLIP: "medium",
            ModelType.FOOD_RECOGNITION: "medium",
            ModelType.YOLO_LARGE: "high",
            ModelType.ACTION_RECOGNITION: "high",
            ModelType.EFFICIENTDET: "high",
            ModelType.GPT4V: "api",
            ModelType.CLAUDE_VISION: "api",
        }
    
    def detect_content_category(self, 
                               speech_text: str,
                               ocr_text: str,
                               caption_text: str,
                               content_type: Optional[str] = None) -> ContentCategory:
        """
        Detect the content category from extracted text.
        
        Args:
            speech_text: Transcribed speech from video
            ocr_text: OCR text from video frames
            caption_text: Video caption text
            content_type: Pre-classified content type (if available)
            
        Returns:
            Detected ContentCategory
        """
        # Combine all text for analysis
        all_text = f"{speech_text} {ocr_text} {caption_text}".lower()
        
        # Check for pre-classified content type
        if content_type:
            ct_lower = content_type.lower()
            if any(food_term in ct_lower for food_term in ["food", "restaurant", "cafe", "bakery"]):
                return ContentCategory.FOOD
            elif any(travel_term in ct_lower for travel_term in ["travel", "place", "hotel", "destination"]):
                return ContentCategory.TRAVEL
            elif any(shopping_term in ct_lower for shopping_term in ["shopping", "product", "review", "haul"]):
                return ContentCategory.SHOPPING
            elif any(fitness_term in ct_lower for fitness_term in ["fitness", "workout", "exercise", "yoga"]):
                return ContentCategory.FITNESS
            elif any(educational_term in ct_lower for educational_term in ["tutorial", "how-to", "lesson", "educational"]):
                return ContentCategory.EDUCATIONAL
            elif any(entertainment_term in ct_lower for entertainment_term in ["entertainment", "funny", "comedy", "meme"]):
                return ContentCategory.ENTERTAINMENT
        
        # Define keyword sets for each category
        food_keywords = ["food", "restaurant", "dish", "meal", "breakfast", "lunch", 
                         "dinner", "cafe", "eat", "cuisine", "delicious", "tasty", 
                         "chef", "menu", "cooking", "recipe", "baking", "dessert"]
                        
        travel_keywords = ["hotel", "resort", "travel", "visit", "vacation", "landmark", 
                           "sightseeing", "tour", "destination", "museum", "gallery", 
                           "attraction", "accommodation", "stay", "location"]
                         
        shopping_keywords = ["product", "buy", "purchase", "shopping", "store", "shop", 
                            "mall", "amazon", "review", "haul", "unboxing", "trying", 
                            "testing", "recommendation"]
        
        fitness_keywords = ["workout", "exercise", "fitness", "gym", "training", 
                            "yoga", "cardio", "strength", "running", "swimming", 
                            "cycling", "hiking", "sport", "active", "health"]
        
        educational_keywords = ["learn", "tutorial", "how to", "explain", "teach", 
                                "lesson", "education", "course", "class", "tip", 
                                "guide", "instruction", "demonstration", "DIY"]
        
        entertainment_keywords = ["funny", "comedy", "joke", "laugh", "meme", 
                                  "entertainment", "sketch", "prank", "dance", 
                                  "song", "music", "movie", "show", "performance"]
        
        # Count keyword matches for each category
        food_score = sum(1 for keyword in food_keywords if keyword in all_text)
        travel_score = sum(1 for keyword in travel_keywords if keyword in all_text)
        shopping_score = sum(1 for keyword in shopping_keywords if keyword in all_text)
        fitness_score = sum(1 for keyword in fitness_keywords if keyword in all_text)
        educational_score = sum(1 for keyword in educational_keywords if keyword in all_text)
        entertainment_score = sum(1 for keyword in entertainment_keywords if keyword in all_text)
        
        # Find the highest scoring category
        scores = {
            ContentCategory.FOOD: food_score,
            ContentCategory.TRAVEL: travel_score,
            ContentCategory.SHOPPING: shopping_score,
            ContentCategory.FITNESS: fitness_score,
            ContentCategory.EDUCATIONAL: educational_score,
            ContentCategory.ENTERTAINMENT: entertainment_score,
        }
        
        max_category = max(scores.items(), key=lambda x: x[1])
        
        # If the highest score is 0 or very low, return UNKNOWN
        if max_category[1] < 2:
            return ContentCategory.UNKNOWN
            
        return max_category[0]
    
    def select_models(self, 
                     content_category: ContentCategory,
                     video_stats: Optional[Dict[str, Any]] = None) -> List[ModelType]:
        """
        Select the appropriate models based on content category and resource limits.
        
        Args:
            content_category: Detected content category
            video_stats: Optional video statistics (resolution, duration, etc.)
            
        Returns:
            List of ModelType enums to apply
        """
        # Base models that are suitable for all categories (lightweight)
        base_models = [ModelType.YOLO_TINY, ModelType.SCENE_CLASSIFICATION]
        
        # Resource-aware model selection
        if self.resource_limit == "low":
            # Only use lightweight models on low-resource systems
            selected_models = [model for model in base_models 
                              if self.model_resource_usage.get(model) == "low"]
            
            # If no food_recognition is available, use YOLO_TINY for food detection
            if content_category == ContentCategory.FOOD:
                if ModelType.YOLO_TINY not in selected_models:
                    selected_models.append(ModelType.YOLO_TINY)
                    
        elif self.resource_limit == "medium":
            # Start with base models
            selected_models = base_models.copy()
            
            # Add category-specific models
            if content_category == ContentCategory.FOOD:
                selected_models.append(ModelType.FOOD_RECOGNITION)
                
            elif content_category == ContentCategory.TRAVEL:
                # CLIP is good for scene understanding
                selected_models.append(ModelType.CLIP)
                
            elif content_category == ContentCategory.SHOPPING:
                # Use YOLO_MEDIUM for better product detection
                if ModelType.YOLO_TINY in selected_models:
                    selected_models.remove(ModelType.YOLO_TINY)
                selected_models.append(ModelType.YOLO_MEDIUM)
                
            elif content_category in [ContentCategory.FITNESS, ContentCategory.ENTERTAINMENT]:
                # Add CLIP for fitness/entertainment content
                selected_models.append(ModelType.CLIP)
                
        else:  # High resource availability
            # Start with base models
            selected_models = base_models.copy()
            
            # Add category-specific models
            if content_category == ContentCategory.FOOD:
                selected_models.append(ModelType.FOOD_RECOGNITION)
                # Replace YOLO_TINY with YOLO_MEDIUM for better detection
                if ModelType.YOLO_TINY in selected_models:
                    selected_models.remove(ModelType.YOLO_TINY)
                selected_models.append(ModelType.YOLO_MEDIUM)
                
            elif content_category == ContentCategory.TRAVEL:
                # Use CLIP for scene understanding and YOLO_MEDIUM for better object detection
                selected_models.append(ModelType.CLIP)
                if ModelType.YOLO_TINY in selected_models:
                    selected_models.remove(ModelType.YOLO_TINY)
                selected_models.append(ModelType.YOLO_MEDIUM)
                
            elif content_category == ContentCategory.SHOPPING:
                # Use YOLO_MEDIUM for better product detection
                if ModelType.YOLO_TINY in selected_models:
                    selected_models.remove(ModelType.YOLO_TINY)
                selected_models.append(ModelType.YOLO_MEDIUM)
                selected_models.append(ModelType.CLIP)
                
            elif content_category == ContentCategory.FITNESS:
                # Use action recognition for fitness content
                selected_models.append(ModelType.ACTION_RECOGNITION)
                selected_models.append(ModelType.CLIP)
                
            elif content_category == ContentCategory.ENTERTAINMENT:
                # Use CLIP and action recognition for entertainment
                selected_models.append(ModelType.CLIP)
                selected_models.append(ModelType.ACTION_RECOGNITION)
                
            elif content_category == ContentCategory.EDUCATIONAL:
                # Use CLIP for educational content
                selected_models.append(ModelType.CLIP)
                
            elif content_category == ContentCategory.UNKNOWN:
                # For unknown content, use a broader selection
                selected_models.append(ModelType.CLIP)
                if ModelType.YOLO_TINY in selected_models:
                    selected_models.remove(ModelType.YOLO_TINY)
                selected_models.append(ModelType.YOLO_MEDIUM)
        
        # Consider API models if enabled and appropriate
        if self.api_enabled:
            # Add API models based on content complexity
            if content_category == ContentCategory.UNKNOWN:
                # For ambiguous content, use Claude Vision for better understanding
                selected_models.append(ModelType.CLAUDE_VISION)
            elif content_category in [ContentCategory.TRAVEL, ContentCategory.FOOD]:
                # These categories often benefit from more sophisticated visual understanding
                selected_models.append(ModelType.GPT4V)
                
        # Remove duplicates while preserving order
        seen = set()
        final_models = []
        for model in selected_models:
            if model not in seen:
                seen.add(model)
                final_models.append(model)
                
        return final_models


def parse_video_for_model_selection(video_path: pathlib.Path) -> Dict[str, Any]:
    """
    Parse basic video statistics for model selection.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video statistics
    """
    try:
        import cv2
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": "Could not open video file"}
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Check video resolution category
        if width * height <= 360 * 640:
            resolution_category = "low"
        elif width * height <= 720 * 1280:
            resolution_category = "medium"
        else:
            resolution_category = "high"
            
        # Release video capture
        cap.release()
        
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "resolution_category": resolution_category
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    """Test the model selector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the model selector")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--content", type=str, default="unknown", help="Content category")
    parser.add_argument("--text", type=str, help="Sample text for classification")
    parser.add_argument("--resource", type=str, default="auto", choices=["low", "medium", "high", "auto"], help="Resource limit")
    parser.add_argument("--api", action="store_true", help="Enable API models")
    
    args = parser.parse_args()
    
    # Create model selector
    selector = ModelSelector(resource_limit=args.resource, api_enabled=args.api)
    
    # Parse sample text if provided
    if args.text:
        content_category = selector.detect_content_category(args.text, "", "")
        print(f"Detected content category: {content_category}")
    else:
        # Use provided content category
        try:
            content_category = ContentCategory(args.content)
        except ValueError:
            print(f"Invalid content category: {args.content}")
            content_category = ContentCategory.UNKNOWN
    
    # Parse video if provided
    video_stats = None
    if args.video:
        video_path = pathlib.Path(args.video)
        if video_path.exists():
            video_stats = parse_video_for_model_selection(video_path)
            print(f"Video stats: {video_stats}")
        else:
            print(f"Video file not found: {video_path}")
    
    # Select models
    selected_models = selector.select_models(content_category, video_stats)
    
    print(f"Selected models for {content_category.value} content:")
    for model in selected_models:
        print(f"  - {model.value}")
    
    # Print resource info
    print(f"\nResource allocation:")
    print(f"  Device: {selector.device}")
    print(f"  Resource limit: {selector.resource_limit}")
    print(f"  API enabled: {selector.api_enabled}")


if __name__ == "__main__":
    main()