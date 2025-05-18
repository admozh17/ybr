#!/usr/bin/env python3
"""
Video recognition module that intelligently selects and applies the appropriate 
recognition models based on video content and existing extraction data.
"""

import os
import cv2
import time
import numpy as np
import pathlib
import torch
import concurrent.futures
from typing import Dict, List, Any, Tuple, Optional, Union
from enum import Enum
import json
import tempfile
import subprocess
from tqdm import tqdm

# Optional imports - only loaded when needed
CLIP_LOADED = False
YOLO_LOADED = False
EFFICIENTDET_LOADED = False
FOOD_MODEL_LOADED = False


class ModelType(Enum):
    """Enumeration of available recognition models."""
    YOLO_TINY = "yolo_tiny"
    YOLO_MEDIUM = "yolo_medium"
    YOLO_LARGE = "yolo_large"
    EFFICIENTDET = "efficientdet"
    CLIP = "clip"
    FOOD_RECOGNITION = "food_recognition"
    SCENE_CLASSIFICATION = "scene_classification"
    ACTION_RECOGNITION = "action_recognition"
    GPT4V = "gpt4v"
    CLAUDE_VISION = "claude_vision"


class RecognitionResult:
    """Container for recognition results."""
    def __init__(self, model_type: ModelType, confidence: float = 0.0):
        self.model_type = model_type
        self.confidence = confidence
        self.objects: List[Dict[str, Any]] = []
        self.scene_categories: List[Dict[str, str]] = []
        self.food_items: List[Dict[str, Any]] = []
        self.activity_types: List[Dict[str, float]] = []
        self.raw_output: Dict[str, Any] = {}
        self.processing_time: float = 0.0
        self.frames_processed: int = 0

    def add_object(self, label: str, confidence: float, bbox: Optional[List[int]] = None):
        """Add a detected object."""
        self.objects.append({
            "label": label,
            "confidence": confidence,
            "bbox": bbox
        })

    def add_scene_category(self, category: str, confidence: float):
        """Add a detected scene category."""
        self.scene_categories.append({
            "category": category,
            "confidence": confidence
        })

    def add_food_item(self, name: str, confidence: float, bbox: Optional[List[int]] = None):
        """Add a detected food item."""
        self.food_items.append({
            "name": name,
            "confidence": confidence,
            "bbox": bbox
        })

    def add_activity_type(self, activity: str, confidence: float):
        """Add a detected activity type."""
        self.activity_types.append({
            "activity": activity,
            "confidence": confidence
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "model_type": self.model_type.value,
            "confidence": self.confidence,
            "objects": self.objects,
            "scene_categories": self.scene_categories,
            "food_items": self.food_items,
            "activity_types": self.activity_types,
            "processing_time": self.processing_time,
            "frames_processed": self.frames_processed
        }

    def __str__(self) -> str:
        return f"RecognitionResult(model={self.model_type.value}, confidence={self.confidence:.2f}, objects={len(self.objects)}, scenes={len(self.scene_categories)}, foods={len(self.food_items)}, activities={len(self.activity_types)})"


class VideoRecognitionAgent:
    """
    Agent for intelligently selecting and applying video recognition models
    based on content type and existing extraction data.
    """
    
    def __init__(self, 
                 use_gpu: bool = True, 
                 max_frames: int = 10, 
                 confidence_threshold: float = 0.5,
                 enable_api_models: bool = True):
        """
        Initialize the video recognition agent.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            max_frames: Maximum number of frames to process per video
            confidence_threshold: Minimum confidence threshold for detections
            enable_api_models: Whether to enable API-based models (GPT-4V, Claude Vision)
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.max_frames = max_frames
        self.confidence_threshold = confidence_threshold
        self.enable_api_models = enable_api_models
        
        # Track models loaded to avoid reloading
        self.loaded_models = {}
        
        # Model selection history for adaptive decision making
        self.model_selection_history = []
        
        print(f"🎥 Initialized VideoRecognitionAgent with device={self.device}, max_frames={max_frames}")
        if self.use_gpu:
            try:
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            except Exception as e:
                print(f"   CUDA available but error getting device info: {e}")

    def _extract_frames(self, video_path: pathlib.Path, num_frames: int = None) -> List[np.ndarray]:
        """
        Extract frames from video for processing.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (None for automatic sampling)
            
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
        
        # Calculate frame sampling interval
        if frame_count <= num_frames:
            # Use all frames if fewer than requested
            sample_indices = list(range(frame_count))
        else:
            # Sample frames evenly across the video
            sample_indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
        
        # Extract frames
        frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                
        cap.release()
        
        print(f"📊 Extracted {len(frames)} frames from video (duration: {duration:.1f}s, {frame_count} total frames)")
        return frames

    def _select_model(self, 
                     content_type: str, 
                     speech_text: str, 
                     ocr_text: str,
                     caption_text: str) -> List[ModelType]:
        """
        Intelligently select which recognition models to apply based on
        content classification and available text.
        
        Args:
            content_type: Classification of video content (if available)
            speech_text: Transcribed speech from video
            ocr_text: OCR text from video frames
            caption_text: Video caption text
            
        Returns:
            List of ModelType enums to apply
        """
        # Combine all text for analysis
        all_text = f"{speech_text} {ocr_text} {caption_text}".lower()
        
        # Food-related keywords
        food_keywords = ["food", "restaurant", "dish", "meal", "breakfast", "lunch", 
                         "dinner", "cafe", "eat", "cuisine", "delicious", "tasty", 
                         "chef", "menu", "cooking", "recipe", "baking", "dessert"]
                        
        # Travel/Place keywords
        place_keywords = ["hotel", "resort", "travel", "visit", "vacation", "landmark", 
                          "sightseeing", "tour", "destination", "museum", "gallery", 
                          "attraction", "accommodation", "stay", "location"]
                         
        # Shopping/product keywords
        product_keywords = ["product", "buy", "purchase", "shopping", "store", "shop", 
                            "mall", "amazon", "review", "haul", "unboxing", "trying", 
                            "testing", "recommendation"]
        
        # Activity keywords
        activity_keywords = ["hiking", "swimming", "surfing", "skiing", "workout", 
                             "exercise", "fitness", "yoga", "dance", "performance", 
                             "concert", "festival", "event", "party", "celebration"]
        
        # Count keyword matches
        food_score = sum(1 for keyword in food_keywords if keyword in all_text)
        place_score = sum(1 for keyword in place_keywords if keyword in all_text)
        product_score = sum(1 for keyword in product_keywords if keyword in all_text)
        activity_score = sum(1 for keyword in activity_keywords if keyword in all_text)
        
        # Default models to run - start with the lightest models
        selected_models = [ModelType.YOLO_TINY, ModelType.SCENE_CLASSIFICATION]
        
        # Adjust based on content detection
        if content_type:
            content_type = content_type.lower()
            if "food" in content_type or "restaurant" in content_type or food_score >= 3:
                selected_models.append(ModelType.FOOD_RECOGNITION)
                # If strongly food-related, replace YOLO_TINY with YOLO_MEDIUM for better detection
                if food_score >= 5:
                    selected_models.remove(ModelType.YOLO_TINY)
                    selected_models.append(ModelType.YOLO_MEDIUM)
                    
            if "travel" in content_type or "place" in content_type or place_score >= 3:
                # Use CLIP for better scene understanding
                if ModelType.YOLO_TINY in selected_models:
                    selected_models.remove(ModelType.YOLO_TINY)
                selected_models.append(ModelType.CLIP)
                
            if "activity" in content_type or activity_score >= 3:
                selected_models.append(ModelType.ACTION_RECOGNITION)
                
            if "product" in content_type or "review" in content_type or product_score >= 3:
                # For product reviews, YOLO_MEDIUM is better at detecting objects
                if ModelType.YOLO_TINY in selected_models:
                    selected_models.remove(ModelType.YOLO_TINY)
                selected_models.append(ModelType.YOLO_MEDIUM)
        
        # Check for advanced or ambiguous cases that might benefit from API models
        complex_case = (
            len(all_text) < 50 or  # Very little text to work with
            food_score + place_score + product_score + activity_score < 2  # Unclear content
        )
        
        if complex_case and self.enable_api_models:
            if not content_type or "unknown" in content_type.lower():
                # For ambiguous content, use Claude Vision for its ability to understand context
                selected_models.append(ModelType.CLAUDE_VISION)
        
        # Track selection for future optimization
        self.model_selection_history.append({
            "content_type": content_type,
            "text_length": len(all_text),
            "food_score": food_score,
            "place_score": place_score,
            "product_score": product_score,
            "activity_score": activity_score,
            "selected_models": [model.value for model in selected_models]
        })
        
        print(f"🤖 Selected models: {[model.value for model in selected_models]}")
        return selected_models

    def _load_yolo_model(self, size: str = "tiny"):
        """Load YOLO model with specified size."""
        global YOLO_LOADED
        
        if f"yolo_{size}" in self.loaded_models:
            return self.loaded_models[f"yolo_{size}"]
            
        try:
            import torch
            # Fix: Map size to correct model name
            if size == "tiny":
                model_name = "yolov8n"  # Use nano model for "tiny"
            elif size == "medium":
                model_name = "yolov8m"
            elif size == "large":
                model_name = "yolov8l"
            else:
                model_name = "yolov8n"  # Default to nano
                
            from ultralytics import YOLO
            model = YOLO(model_name)
            model.to(self.device)
            
            self.loaded_models[f"yolo_{size}"] = model
            YOLO_LOADED = True
            print(f"✅ Loaded YOLO {size} model ({model_name})")
            return model
        except ImportError as e:
            print(f"❌ Failed to load YOLO model: {e}. Make sure ultralytics is installed.")
            return None
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            return None

    def _load_clip_model(self):
        """Load CLIP model."""
        global CLIP_LOADED
        
        if "clip" in self.loaded_models:
            return self.loaded_models["clip"]
            
        try:
            import torch
            try:
                import clip
            except ImportError:
                try:
                    from PIL import Image
                    print("Installing CLIP via pip...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
                    import clip
                except Exception as e:
                    print(f"❌ Failed to install CLIP: {e}")
                    return None
                
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            self.loaded_models["clip"] = (model, preprocess)
            CLIP_LOADED = True
            print(f"✅ Loaded CLIP model")
            return model, preprocess
        except ImportError as e:
            print(f"❌ Failed to load CLIP model: {e}. Make sure clip is installed.")
            return None
        except Exception as e:
            print(f"❌ Error loading CLIP model: {e}")
            return None

    def _load_food_recognition_model(self):
        """Load food recognition model."""
        global FOOD_MODEL_LOADED
        
        if "food_recognition" in self.loaded_models:
            return self.loaded_models["food_recognition"]
            
        try:
            # Using YOLO with food dataset like Food-101
            import torch
            from ultralytics import YOLO
            # First check if we have a custom food model
            if os.path.exists("models/yolov8m-food.pt"):
                model = YOLO("models/yolov8m-food.pt")
            else:
                # Fall back to regular YOLO model
                model = YOLO("yolov8m")
            model.to(self.device)
            
            self.loaded_models["food_recognition"] = model
            FOOD_MODEL_LOADED = True
            print(f"✅ Loaded food recognition model")
            return model
        except ImportError as e:
            print(f"❌ Failed to load food recognition model: {e}")
            return None
        except Exception as e:
            print(f"❌ Error loading food recognition model: {e}")
            return None

    def _load_scene_classification_model(self):
        """Load scene classification model."""
        if "scene_classification" in self.loaded_models:
            return self.loaded_models["scene_classification"]
            
        try:
            import torch
            try:
                import timm
            except ImportError:
                try:
                    print("Installing timm via pip...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
                    import timm
                except Exception as e:
                    print(f"❌ Failed to install timm: {e}")
                    return None
                    
            # Use a lightweight model like MobileNetV3 pretrained on Places365 or ImageNet
            try:
                model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=365)
            except Exception as e:
                print(f"Error loading mobilenetv3 with Places365: {e}")
                # Fallback to a more common model
                model = timm.create_model('resnet18', pretrained=True)
                
            model.to(self.device)
            
            # Load places category names
            scene_categories = []
            try:
                # Try to load categories from file
                if os.path.exists("models/places365_categories.txt"):
                    with open("models/places365_categories.txt", "r") as f:
                        scene_categories = [line.strip() for line in f.readlines()]
                else:
                    # Default to common scene categories
                    scene_categories = ["restaurant", "cafe", "hotel", "beach", "mountain", 
                                        "park", "forest", "museum", "shopping_mall", "street",
                                        "indoor", "outdoor", "natural", "urban", "water", 
                                        "transportation", "architecture", "food", "sports"]
            except Exception as e:
                print(f"Could not load scene categories: {e}")
                scene_categories = ["indoor", "outdoor", "natural", "urban", "water", "transportation"]
            
            self.loaded_models["scene_classification"] = (model, scene_categories)
            print(f"✅ Loaded scene classification model")
            return model, scene_categories
        except ImportError as e:
            print(f"❌ Failed to load scene classification model: {e}. Make sure timm is installed.")
            return None
        except Exception as e:
            print(f"❌ Error loading scene classification model: {e}")
            return None

    def _load_action_recognition_model(self):
        """Load action recognition model."""
        if "action_recognition" in self.loaded_models:
            return self.loaded_models["action_recognition"]
            
        try:
            import torch
            try:
                import pytorchvideo.models.hub as hub
            except ImportError:
                print("❌ Failed to import pytorchvideo. Action recognition disabled.")
                return None
                
            # Load a lightweight model
            try:
                model = hub.slowfast_r50(pretrained=True)
                model.to(self.device)
            except Exception as e:
                print(f"❌ Error loading SlowFast model: {e}")
                return None
            
            # Load action categories
            action_categories = []
            try:
                # Try to load categories from file
                if os.path.exists("models/kinetics400_categories.txt"):
                    with open("models/kinetics400_categories.txt", "r") as f:
                        action_categories = [line.strip() for line in f.readlines()]
                else:
                    # Default to common action categories
                    action_categories = ["eating", "drinking", "walking", "running", "swimming", 
                                         "dancing", "cooking", "shopping", "driving", "playing",
                                         "exercising", "traveling", "relaxing", "working", "socializing"]
            except Exception as e:
                print(f"Could not load action categories: {e}")
                action_categories = ["eating", "exercising", "traveling", "relaxing", "working", "socializing"]
            
            self.loaded_models["action_recognition"] = (model, action_categories)
            print(f"✅ Loaded action recognition model")
            return model, action_categories
        except ImportError as e:
            print(f"❌ Failed to load action recognition model: {e}. Make sure pytorchvideo is installed.")
            return None
        except Exception as e:
            print(f"❌ Error loading action recognition model: {e}")
            return None

    async def _call_gpt4v_api(self, frames: List[np.ndarray], prompt: str) -> Dict:
        """Call GPT-4 Vision API with frames."""
        try:
            import openai
            import base64
            from io import BytesIO
            from PIL import Image
            
            # Select key frames to analyze (first, middle, last)
            if len(frames) > 3:
                analysis_frames = [frames[0], frames[len(frames)//2], frames[-1]]
            else:
                analysis_frames = frames
                
            # Encode frames as base64
            encoded_images = []
            for frame in analysis_frames:
                # Convert from BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                buffer = BytesIO()
                pil_image.save(buffer, format="JPEG")
                encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                encoded_images.append(encoded_image)
            
            # Build messages with context
            messages = [
                {"role": "system", "content": "You are a video analysis assistant. Analyze the frames from this video and provide detailed information about what you see."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in encoded_images]
                ]}
            ]
            
            # Call API
            response = await openai.ChatCompletion.acreate(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=500
            )
            
            return {"response": response.choices[0].message.content}
        except Exception as e:
            print(f"❌ Error calling GPT-4V API: {e}")
            return {"error": str(e)}

    async def _call_claude_vision_api(self, frames: List[np.ndarray], prompt: str) -> Dict:
        """Call Claude Vision API with frames."""
        try:
            import anthropic
            import base64
            from io import BytesIO
            from PIL import Image
            
            # Select key frames to analyze (first, middle, last)
            if len(frames) > 3:
                analysis_frames = [frames[0], frames[len(frames)//2], frames[-1]]
            else:
                analysis_frames = frames
                
            # Encode frames as base64
            encoded_images = []
            for frame in analysis_frames:
                # Convert from BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                buffer = BytesIO()
                pil_image.save(buffer, format="JPEG")
                encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                encoded_images.append(
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded_image}}
                )
            
            # Create client
            client = anthropic.Anthropic()
            
            # Build message
            message = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        *encoded_images
                    ]}
                ]
            )
            
            return {"response": message.content[0].text}
        except Exception as e:
            print(f"❌ Error calling Claude Vision API: {e}")
            return {"error": str(e)}

    def _run_yolo_detection(self, frames: List[np.ndarray], size: str = "tiny") -> RecognitionResult:
        """Run YOLO object detection on frames."""
        start_time = time.time()
        result = RecognitionResult(
            model_type=ModelType.YOLO_TINY if size == "tiny" else 
                       ModelType.YOLO_MEDIUM if size == "medium" else 
                       ModelType.YOLO_LARGE
        )
        
        try:
            # Load model
            model = self._load_yolo_model(size)
            if model is None:
                return result
                
            # Process frames
            processed_count = 0
            for frame in frames:
                # Run detection
                try:
                    detections = model(frame)[0]
                    
                    # Process results
                    for detection in detections.boxes.data.tolist():
                        x1, y1, x2, y2, conf, class_id = detection
                        if conf >= self.confidence_threshold:
                            label = detections.names[int(class_id)]
                            result.add_object(label, float(conf), [int(x1), int(y1), int(x2), int(y2)])
                except Exception as e:
                    print(f"⚠️ Error processing frame with YOLO: {e}")
                    continue
                        
                processed_count += 1
                
            # Aggregate results
            # Group by label and take average confidence
            label_counts = {}
            for obj in result.objects:
                label = obj["label"]
                if label not in label_counts:
                    label_counts[label] = {"count": 0, "conf_sum": 0}
                label_counts[label]["count"] += 1
                label_counts[label]["conf_sum"] += obj["confidence"]
            
            # Sort by count and confidence
            sorted_labels = sorted(
                label_counts.items(), 
                key=lambda x: (x[1]["count"], x[1]["conf_sum"] / x[1]["count"]), 
                reverse=True
            )
            
            # Set overall confidence as average of top 3 detected objects
            if sorted_labels:
                top_labels = sorted_labels[:min(3, len(sorted_labels))]
                avg_conf = sum(label_info["conf_sum"] / label_info["count"] for _, label_info in top_labels) / len(top_labels)
                result.confidence = avg_conf
            
            # Complete result
            result.processing_time = time.time() - start_time
            result.frames_processed = processed_count
            
            print(f"🔍 YOLO {size} detected {len(sorted_labels)} object types in {result.processing_time:.2f}s")
            if sorted_labels:
                top_objects = []
                for label, info in sorted_labels[:3]:
                    count = info["count"]
                    avg_conf = info["conf_sum"] / count
                    top_objects.append(f"{label} ({count}x, {avg_conf:.2f})")
                print(f"   Top objects: {', '.join(top_objects)}")
                
            return result
        except Exception as e:
            print(f"❌ Error running YOLO detection: {e}")
            result.processing_time = time.time() - start_time
            return result

    def _run_clip_analysis(self, frames: List[np.ndarray]) -> RecognitionResult:
        """Run CLIP analysis on frames."""
        start_time = time.time()
        result = RecognitionResult(model_type=ModelType.CLIP)
        
        try:
            # Load model
            model_data = self._load_clip_model()
            if model_data is None:
                return result
                
            model, preprocess = model_data
            
            # Import PIL here for CLIP preprocessing
            from PIL import Image
            
            # Candidate categories for scene/place classification
            categories = [
                "restaurant", "cafe", "bar", "coffee shop", 
                "hotel", "resort", "apartment", "house",
                "beach", "mountain", "forest", "park", "lake", "ocean",
                "museum", "gallery", "theater", "stadium", "concert hall",
                "store", "shop", "mall", "market", "boutique",
                "gym", "fitness center", "spa", "wellness center",
                "airport", "train station", "bus station", "harbor", "port",
                "street", "road", "highway", "path", "trail",
                "school", "university", "library", "bookstore",
                "hospital", "clinic", "doctor's office", "pharmacy",
                "office", "workspace", "coworking space",
                "farm", "garden", "vineyard", "orchard",
                "temple", "church", "mosque", "synagogue", "religious building",
                "castle", "palace", "monument", "historic site", "landmark"
            ]
            
            import clip
            # Convert to tensor
            text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(self.device)
            
            processed_count = 0
            category_scores = {category: 0.0 for category in categories}
            
            # Process frames
            for frame in frames:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Preprocess image
                image = preprocess(Image.fromarray(rgb_frame)).unsqueeze(0).to(self.device)
                
                # Calculate features
                with torch.no_grad():
                    image_features = model.encode_image(image)
                    text_features = model.encode_text(text)
                    
                    # Normalize
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarity
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
                
                # Update scores
                for i, category in enumerate(categories):
                    category_scores[category] += float(similarity[i])
                
                processed_count += 1
            
            # Average scores across frames
            if processed_count > 0:
                for category in category_scores:
                    category_scores[category] /= processed_count
            
            # Sort by score
            sorted_categories = sorted(
                category_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Add top categories to result
            for category, score in sorted_categories[:10]:  # Top 10 categories
                if score >= self.confidence_threshold:
                    result.add_scene_category(category, score)
            
            # Set overall confidence
            if result.scene_categories:
                result.confidence = sum(cat["confidence"] for cat in result.scene_categories[:3]) / min(3, len(result.scene_categories))
            
            # Complete result
            result.processing_time = time.time() - start_time
            result.frames_processed = processed_count
            
            print(f"🔍 CLIP analyzed {processed_count} frames in {result.processing_time:.2f}s")
            if result.scene_categories:
                print(f"   Top scenes: {', '.join([f'{cat['category']} ({cat['confidence']:.2f})' for cat in result.scene_categories[:3]])}")
                
            return result
        except Exception as e:
            print(f"❌ Error running CLIP analysis: {e}")
            result.processing_time = time.time() - start_time
            return result

    def _run_food_recognition(self, frames: List[np.ndarray]) -> RecognitionResult:
        """Run food recognition on frames."""
        start_time = time.time()
        result = RecognitionResult(model_type=ModelType.FOOD_RECOGNITION)
        
        try:
            # Load model
            model = self._load_food_recognition_model()
            if model is None:
                return result
                
            # Food categories to focus on (can be expanded)
            food_categories = [
                "pizza", "burger", "sandwich", "hotdog", "taco", "burrito", "pasta", "noodles",
                "rice", "salad", "soup", "steak", "fish", "chicken", "pork", "beef", "lamb",
                "cake", "pastry", "cookie", "pie", "ice cream", "donut", "croissant", "bread",
                "fruit", "vegetable", "sushi", "seafood", "cheese", "egg", "pancake", "waffle",
                "coffee", "tea", "juice", "smoothie", "wine", "beer", "cocktail"
            ]
            
            processed_count = 0
            for frame in frames:
                try:
                    # Run detection
                    detections = model(frame)[0]
                    
                    # Process results - focus on food items
                    for detection in detections.boxes.data.tolist():
                        x1, y1, x2, y2, conf, class_id = detection
                        if conf >= self.confidence_threshold:
                            label = detections.names[int(class_id)]
                            
                            # Check if it's a food item
                            is_food = any(food_term in label.lower() for food_term in food_categories)
                            if is_food or "food" in label.lower() or "dish" in label.lower():
                                result.add_food_item(label, float(conf), [int(x1), int(y1), int(x2), int(y2)])
                except Exception as e:
                    print(f"⚠️ Error processing frame for food recognition: {e}")
                    continue
                        
                processed_count += 1
                
            # Aggregate results
            # Group by food name and take average confidence
            food_counts = {}
            for food in result.food_items:
                name = food["name"]
                if name not in food_counts:
                    food_counts[name] = {"count": 0, "conf_sum": 0}
                food_counts[name]["count"] += 1
                food_counts[name]["conf_sum"] += food["confidence"]
            
            # Sort by count and confidence
            sorted_foods = sorted(
                food_counts.items(), 
                key=lambda x: (x[1]["count"], x[1]["conf_sum"] / x[1]["count"]), 
                reverse=True
            )
            
            # Set overall confidence
            if sorted_foods:
                top_foods = sorted_foods[:min(3, len(sorted_foods))]
                avg_conf = sum(food_info["conf_sum"] / food_info["count"] for _, food_info in top_foods) / len(top_foods)
                result.confidence = avg_conf
            
            # Complete result
            result.processing_time = time.time() - start_time
            result.frames_processed = processed_count
            
            print(f"🍽️ Food recognition processed {processed_count} frames in {result.processing_time:.2f}s")
            if sorted_foods:
                food_texts = []
                for food, info in sorted_foods[:3]:
                    count = info["count"]
                    avg_conf = info["conf_sum"] / count
                    food_texts.append(f"{food} ({count}x, {avg_conf:.2f})")
                print(f"   Top foods: {', '.join(food_texts)}")
                
            return result
        except Exception as e:
            print(f"❌ Error running food recognition: {e}")
            result.processing_time = time.time() - start_time
            return result

    def _run_scene_classification(self, frames: List[np.ndarray]) -> RecognitionResult:
        """Run scene classification on frames."""
        start_time = time.time()
        result = RecognitionResult(model_type=ModelType.SCENE_CLASSIFICATION)
        
        try:
            # Load model
            model_data = self._load_scene_classification_model()
            if model_data is None:
                return result
                
            model, categories = model_data
            
            # Setup preprocessing
            import torchvision.transforms as transforms
            from PIL import Image
            
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            processed_count = 0
            category_scores = {category: 0.0 for category in categories}
            
            # Process frames
            for frame in frames:
                try:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Preprocess image
                    input_tensor = preprocess(rgb_frame).unsqueeze(0).to(self.device)
                    
                    # Run model
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    
                    # Update scores
                    for i, category in enumerate(categories):
                        if i < len(probabilities):
                            category_scores[category] += float(probabilities[i])
                except Exception as e:
                    print(f"⚠️ Error processing frame for scene classification: {e}")
                    continue
                
                processed_count += 1
            
            # Average scores across frames
            if processed_count > 0:
                for category in category_scores:
                    category_scores[category] /= processed_count
            
            # Sort by score
            sorted_categories = sorted(
                category_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Add top categories to result
            for category, score in sorted_categories[:10]:  # Top 10 categories
                if score >= self.confidence_threshold:
                    result.add_scene_category(category, score)
            
            # Set overall confidence
            if result.scene_categories:
                result.confidence = sum(cat["confidence"] for cat in result.scene_categories[:3]) / min(3, len(result.scene_categories))
            
            # Complete result
            result.processing_time = time.time() - start_time
            result.frames_processed = processed_count
            
            print(f"🏙️ Scene classification processed {processed_count} frames in {result.processing_time:.2f}s")
            if result.scene_categories:
                scene_texts = []
                for cat in result.scene_categories[:3]:
                    category = cat["category"]
                    confidence = cat["confidence"]
                    scene_texts.append(f"{category} ({confidence:.2f})")
                print(f"   Top scenes: {', '.join(scene_texts)}")
                
            return result
        except Exception as e:
            print(f"❌ Error running scene classification: {e}")
            result.processing_time = time.time() - start_time
            return result

    def _run_action_recognition(self, frames: List[np.ndarray]) -> RecognitionResult:
        """Run action recognition on frames."""
        start_time = time.time()
        result = RecognitionResult(model_type=ModelType.ACTION_RECOGNITION)
        
        try:
            # Load model
            model_data = self._load_action_recognition_model()
            if model_data is None:
                return result
                
            model, categories = model_data
            
            # We need a sequence of frames for action recognition
            # If we don't have enough frames, duplicate them
            if len(frames) < 8:
                frames = frames * (8 // len(frames) + 1)
                frames = frames[:8]
            
            # Setup preprocessing
            import torch
            import torchvision.transforms as transforms
            from PIL import Image
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            ])
            
            try:
                # Process frames
                frame_tensors = []
                for frame in frames:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Transform frame
                    frame_tensor = transform(rgb_frame)
                    frame_tensors.append(frame_tensor)
                
                # Stack frames and add batch dimension
                stacked_frames = torch.stack(frame_tensors)
                input_tensor = stacked_frames.unsqueeze(0).to(self.device)
                
                # Run model
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                
                # Get top actions
                topk_values, topk_indices = torch.topk(probabilities, 5)
                
                # Add to result
                for i in range(len(topk_indices)):
                    idx = int(topk_indices[i])
                    score = float(topk_values[i])
                    if idx < len(categories) and score >= self.confidence_threshold:
                        activity = categories[idx]
                        result.add_activity_type(activity, score)
            except Exception as e:
                print(f"⚠️ Error during action recognition inference: {e}")
            
            # Set overall confidence
            if result.activity_types:
                result.confidence = sum(act["confidence"] for act in result.activity_types[:3]) / min(3, len(result.activity_types))
            
            # Complete result
            result.processing_time = time.time() - start_time
            result.frames_processed = len(frames)
            
            print(f"🏃 Action recognition processed {len(frames)} frames in {result.processing_time:.2f}s")
            if result.activity_types:
                activity_texts = []
                for act in result.activity_types[:3]:
                    activity = act["activity"]
                    confidence = act["confidence"]
                    activity_texts.append(f"{activity} ({confidence:.2f})")
                print(f"   Top activities: {', '.join(activity_texts)}")
                
            return result
        except Exception as e:
            print(f"❌ Error running action recognition: {e}")
            result.processing_time = time.time() - start_time
            return result

    async def _run_gpt4v_analysis(self, frames: List[np.ndarray]) -> RecognitionResult:
        """Run GPT-4V analysis on frames."""
        start_time = time.time()
        result = RecognitionResult(model_type=ModelType.GPT4V)
        
        try:
            prompt = """
            Please analyze these frames from a short-form video (like Instagram Reel, TikTok, or YouTube Short).
            
            1. What type of place or establishment is shown? (e.g., restaurant, cafe, hotel, landmark, etc.)
            2. Are there any specific food items or dishes visible? List them.
            3. What is the overall scene or setting? (e.g., indoor dining, beach, city street, etc.)
            4. Is there any activity happening? (e.g., eating, traveling, shopping, etc.)
            5. Are there any notable objects or items that might help identify the place?
            
            Please be specific and detailed in your analysis. Format your response as a JSON object with these keys:
            {
                "place_type": "Type of place/establishment",
                "food_items": ["List", "of", "foods"],
                "scene": "Description of scene/setting",
                "activities": ["List", "of", "activities"],
                "notable_objects": ["List", "of", "objects"]
            }
            """
            
            # Call API
            api_result = await self._call_gpt4v_api(frames, prompt)
            
            # Process response
            if "error" in api_result:
                print(f"Error from GPT-4V API: {api_result['error']}")
                return result
                
            response = api_result["response"]
            
            # Try to extract JSON
            json_data = {}
            try:
                # Look for JSON block
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON without code block
                    json_match = re.search(r'({.*})', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_str = response
                        
                json_data = json.loads(json_str)
            except Exception as e:
                print(f"Could not parse JSON from GPT-4V response: {e}")
                # Fallback: extract information using regex
                try:
                    import re
                    place_type_match = re.search(r'place_type"?\s*:\s*"([^"]+)"', response)
                    food_items_match = re.search(r'food_items"?\s*:\s*\[(.*?)\]', response, re.DOTALL)
                    scene_match = re.search(r'scene"?\s*:\s*"([^"]+)"', response)
                    activities_match = re.search(r'activities"?\s*:\s*\[(.*?)\]', response, re.DOTALL)
                    objects_match = re.search(r'notable_objects"?\s*:\s*\[(.*?)\]', response, re.DOTALL)
                    
                    json_data = {
                        "place_type": place_type_match.group(1) if place_type_match else "",
                        "food_items": re.findall(r'"([^"]+)"', food_items_match.group(1)) if food_items_match else [],
                        "scene": scene_match.group(1) if scene_match else "",
                        "activities": re.findall(r'"([^"]+)"', activities_match.group(1)) if activities_match else [],
                        "notable_objects": re.findall(r'"([^"]+)"', objects_match.group(1)) if objects_match else []
                    }
                except Exception as regex_error:
                    print(f"Regex extraction also failed: {regex_error}")
                    # Create empty structure
                    json_data = {
                        "place_type": "",
                        "food_items": [],
                        "scene": "",
                        "activities": [],
                        "notable_objects": []
                    }
            
            # Add results
            if "place_type" in json_data and json_data["place_type"]:
                result.add_scene_category(json_data["place_type"], 0.95)
                
            if "scene" in json_data and json_data["scene"]:
                result.add_scene_category(json_data["scene"], 0.9)
                
            if "food_items" in json_data and json_data["food_items"]:
                for food in json_data["food_items"]:
                    result.add_food_item(food, 0.9)
                    
            if "activities" in json_data and json_data["activities"]:
                for activity in json_data["activities"]:
                    result.add_activity_type(activity, 0.9)
                    
            if "notable_objects" in json_data and json_data["notable_objects"]:
                for obj in json_data["notable_objects"]:
                    result.add_object(obj, 0.9)
            
            # Add raw output for reference
            result.raw_output = {
                "gpt4v_response": response,
                "parsed_data": json_data
            }
            
            # Set confidence
            result.confidence = 0.9  # GPT-4V results are generally reliable
            
            # Complete result
            result.processing_time = time.time() - start_time
            result.frames_processed = len(frames)
            
            print(f"🤖 GPT-4V analysis completed in {result.processing_time:.2f}s")
            
            return result
        except Exception as e:
            print(f"❌ Error running GPT-4V analysis: {e}")
            result.processing_time = time.time() - start_time
            return result

    async def _run_claude_vision_analysis(self, frames: List[np.ndarray]) -> RecognitionResult:
        """Run Claude Vision analysis on frames."""
        start_time = time.time()
        result = RecognitionResult(model_type=ModelType.CLAUDE_VISION)
        
        try:
            prompt = """
            Please analyze these frames from a short-form video (like Instagram Reel, TikTok, or YouTube Short).
            
            1. What type of place or establishment is shown? (e.g., restaurant, cafe, hotel, landmark, etc.)
            2. Are there any specific food items or dishes visible? List them.
            3. What is the overall scene or setting? (e.g., indoor dining, beach, city street, etc.)
            4. Is there any activity happening? (e.g., eating, traveling, shopping, etc.)
            5. Are there any notable objects or items that might help identify the place?
            
            Please be specific and detailed in your analysis. Format your response as a JSON object with these keys:
            {
                "place_type": "Type of place/establishment",
                "food_items": ["List", "of", "foods"],
                "scene": "Description of scene/setting",
                "activities": ["List", "of", "activities"],
                "notable_objects": ["List", "of", "objects"]
            }
            """
            
            # Call API
            api_result = await self._call_claude_vision_api(frames, prompt)
            
            # Process response
            if "error" in api_result:
                print(f"Error from Claude Vision API: {api_result['error']}")
                return result
                
            response = api_result["response"]
            
            # Try to extract JSON
            json_data = {}
            try:
                # Look for JSON block
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON without code block
                    json_match = re.search(r'({.*})', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_str = response
                        
                json_data = json.loads(json_str)
            except Exception as e:
                print(f"Could not parse JSON from Claude Vision response: {e}")
                # Fallback: extract information using regex
                try:
                    import re
                    place_type_match = re.search(r'place_type"?\s*:\s*"([^"]+)"', response)
                    food_items_match = re.search(r'food_items"?\s*:\s*\[(.*?)\]', response, re.DOTALL)
                    scene_match = re.search(r'scene"?\s*:\s*"([^"]+)"', response)
                    activities_match = re.search(r'activities"?\s*:\s*\[(.*?)\]', response, re.DOTALL)
                    objects_match = re.search(r'notable_objects"?\s*:\s*\[(.*?)\]', response, re.DOTALL)
                    
                    json_data = {
                        "place_type": place_type_match.group(1) if place_type_match else "",
                        "food_items": re.findall(r'"([^"]+)"', food_items_match.group(1)) if food_items_match else [],
                        "scene": scene_match.group(1) if scene_match else "",
                        "activities": re.findall(r'"([^"]+)"', activities_match.group(1)) if activities_match else [],
                        "notable_objects": re.findall(r'"([^"]+)"', objects_match.group(1)) if objects_match else []
                    }
                except Exception as regex_error:
                    print(f"Regex extraction also failed: {regex_error}")
                    # Create empty structure
                    json_data = {
                        "place_type": "",
                        "food_items": [],
                        "scene": "",
                        "activities": [],
                        "notable_objects": []
                    }
            
            # Add results
            if "place_type" in json_data and json_data["place_type"]:
                result.add_scene_category(json_data["place_type"], 0.95)
                
            if "scene" in json_data and json_data["scene"]:
                result.add_scene_category(json_data["scene"], 0.9)
                
            if "food_items" in json_data and json_data["food_items"]:
                for food in json_data["food_items"]:
                    result.add_food_item(food, 0.9)
                    
            if "activities" in json_data and json_data["activities"]:
                for activity in json_data["activities"]:
                    result.add_activity_type(activity, 0.9)
                    
            if "notable_objects" in json_data and json_data["notable_objects"]:
                for obj in json_data["notable_objects"]:
                    result.add_object(obj, 0.9)
            
            # Add raw output for reference
            result.raw_output = {
                "claude_vision_response": response,
                "parsed_data": json_data
            }
            
            # Set confidence
            result.confidence = 0.9  # Claude Vision results are generally reliable
            
            # Complete result
            result.processing_time = time.time() - start_time
            result.frames_processed = len(frames)
            
            print(f"🤖 Claude Vision analysis completed in {result.processing_time:.2f}s")
            
            return result
        except Exception as e:
            print(f"❌ Error running Claude Vision analysis: {e}")
            result.processing_time = time.time() - start_time
            return result

    async def process_video(self, 
                     video_path: pathlib.Path,
                     content_type: Optional[str] = None,
                     speech_text: str = "",
                     ocr_text: str = "",
                     caption_text: str = "") -> Dict[str, Any]:
        """
        Process a video with the appropriate recognition models.
        
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
            frames = self._extract_frames(video_path, self.max_frames)
            if not frames:
                return {"error": "No frames could be extracted from video"}
                
            # Select models
            selected_models = self._select_model(content_type, speech_text, ocr_text, caption_text)
            
            # Initialize results
            results = {}
            
            # Run local models in parallel with ThreadPoolExecutor
            local_models = [model for model in selected_models if model not in [ModelType.GPT4V, ModelType.CLAUDE_VISION]]
            api_models = [model for model in selected_models if model in [ModelType.GPT4V, ModelType.CLAUDE_VISION]]
            
            local_results = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit local model processing tasks
                future_to_model = {}
                
                for model in local_models:
                    if model == ModelType.YOLO_TINY:
                        future = executor.submit(self._run_yolo_detection, frames, "tiny")
                    elif model == ModelType.YOLO_MEDIUM:
                        future = executor.submit(self._run_yolo_detection, frames, "medium")
                    elif model == ModelType.YOLO_LARGE:
                        future = executor.submit(self._run_yolo_detection, frames, "large")
                    elif model == ModelType.CLIP:
                        future = executor.submit(self._run_clip_analysis, frames)
                    elif model == ModelType.FOOD_RECOGNITION:
                        future = executor.submit(self._run_food_recognition, frames)
                    elif model == ModelType.SCENE_CLASSIFICATION:
                        future = executor.submit(self._run_scene_classification, frames)
                    elif model == ModelType.ACTION_RECOGNITION:
                        future = executor.submit(self._run_action_recognition, frames)
                    else:
                        continue
                        
                    future_to_model[future] = model
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_model):
                    model = future_to_model[future]
                    try:
                        result = future.result()
                        local_results[model.value] = result
                    except Exception as e:
                        print(f"Error processing {model.value}: {e}")
            
            # Run API models sequentially (to respect rate limits)
            api_results = {}
            for model in api_models:
                try:
                    if model == ModelType.GPT4V:
                        result = await self._run_gpt4v_analysis(frames)
                    elif model == ModelType.CLAUDE_VISION:
                        result = await self._run_claude_vision_analysis(frames)
                    else:
                        continue
                        
                    api_results[model.value] = result
                except Exception as e:
                    print(f"Error processing {model.value}: {e}")
            
            # Combine results
            all_results = {**local_results, **api_results}
            
            # Convert to response format
            recognition_results = []
            for model_type, result in all_results.items():
                recognition_results.append(result.to_dict())
            
            # Return comprehensive result
            return {
                "recognition_results": recognition_results,
                "processing_time": time.time() - start_time,
                "frames_processed": len(frames),
                "models_used": [model.value for model in selected_models]
            }
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def run_sync(self, 
                video_path: pathlib.Path,
                content_type: Optional[str] = None,
                speech_text: str = "",
                ocr_text: str = "",
                caption_text: str = "") -> Dict[str, Any]:
        """
        Process a video with the appropriate recognition models synchronously.
        
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
            frames = self._extract_frames(video_path, self.max_frames)
            if not frames:
                return {"error": "No frames could be extracted from video"}
                
            # Select models
            selected_models = self._select_model(content_type, speech_text, ocr_text, caption_text)
            
            # Filter out API models in synchronous mode
            selected_models = [model for model in selected_models if model not in [ModelType.GPT4V, ModelType.CLAUDE_VISION]]
            
            # Initialize results
            results = {}
            
            # Run local models in parallel with ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit local model processing tasks
                future_to_model = {}
                
                for model in selected_models:
                    if model == ModelType.YOLO_TINY:
                        future = executor.submit(self._run_yolo_detection, frames, "tiny")
                    elif model == ModelType.YOLO_MEDIUM:
                        future = executor.submit(self._run_yolo_detection, frames, "medium")
                    elif model == ModelType.YOLO_LARGE:
                        future = executor.submit(self._run_yolo_detection, frames, "large")
                    elif model == ModelType.CLIP:
                        future = executor.submit(self._run_clip_analysis, frames)
                    elif model == ModelType.FOOD_RECOGNITION:
                        future = executor.submit(self._run_food_recognition, frames)
                    elif model == ModelType.SCENE_CLASSIFICATION:
                        future = executor.submit(self._run_scene_classification, frames)
                    elif model == ModelType.ACTION_RECOGNITION:
                        future = executor.submit(self._run_action_recognition, frames)
                    else:
                        continue
                        
                    future_to_model[future] = model
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_model):
                    model = future_to_model[future]
                    try:
                        result = future.result()
                        results[model.value] = result
                    except Exception as e:
                        print(f"Error processing {model.value}: {e}")
            
            # Convert to response format
            recognition_results = []
            for model_type, result in results.items():
                recognition_results.append(result.to_dict())
            
            # Return comprehensive result
            return {
                "recognition_results": recognition_results,
                "processing_time": time.time() - start_time,
                "frames_processed": len(frames),
                "models_used": [model.value for model in selected_models]
            }
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def run_minimal_recognition(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Run minimal recognition with just basic YOLO and scene classification.
        This is designed to be lightweight and work without additional dependencies.
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary with recognition results
        """
        start_time = time.time()
        
        # Run scene classification first as it's more likely to succeed
        scene_result = self._run_scene_classification(frames)
        
        # Try to run YOLO detection
        try:
            yolo_result = self._run_yolo_detection(frames, "tiny")
        except Exception as e:
            print(f"⚠️ YOLO detection failed in minimal recognition: {e}")
            # Create empty YOLO result
            yolo_result = RecognitionResult(model_type=ModelType.YOLO_TINY)
            yolo_result.processing_time = 0.0
            yolo_result.frames_processed = len(frames)
        
        # Combine results
        recognition_results = []
        if yolo_result:
            recognition_results.append(yolo_result.to_dict())
        if scene_result:
            recognition_results.append(scene_result.to_dict())
        
        # Return minimal result
        return {
            "recognition_results": recognition_results,
            "processing_time": time.time() - start_time,
            "frames_processed": len(frames),
            "models_used": ["yolo_tiny", "scene_classification"]
        }
        