#!/usr/bin/env python3
"""Vision worker microservice for parallel video recognition."""

import os
import tempfile
import pathlib
import json
import time
from typing import List, Dict, Any, Tuple, Optional, Union
import asyncio
import aiohttp

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import cv2
import torch

# Initialize FastAPI
app = FastAPI(title="Vision Microservice")

# Global model cache
vision_models = {}

class TaskSpec(BaseModel):
    """Model for vision task specification."""
    model_types: List[str] = Field(..., description="List of model types to run")
    frame_urls: List[str] = Field(..., description="URLs of frames to process")
    confidence: float = Field(default=0.5, description="Confidence threshold")

class VisionResponse(BaseModel):
    """Response model for vision processing."""
    results: List[Dict[str, Any]] = Field(..., description="Recognition results")
    processing_time: float = Field(..., description="Processing time in seconds")
    frames_processed: int = Field(..., description="Number of frames processed")
    models_used: List[str] = Field(..., description="Models used")

class ModelType:
    """Constants for supported model types."""
    YOLO_TINY = "yolo_tiny"
    YOLO_MEDIUM = "yolo_medium"
    YOLO_LARGE = "yolo_large"
    SCENE_CLASSIFICATION = "scene_classification"
    GDINO = "gdino"
    FOOD_RECOGNITION = "food_recognition"
    
    @staticmethod
    def all_models():
        return [
            ModelType.YOLO_TINY,
            ModelType.YOLO_MEDIUM,
            ModelType.YOLO_LARGE,
            ModelType.SCENE_CLASSIFICATION,
            ModelType.GDINO,
            ModelType.FOOD_RECOGNITION
        ]

# Model loading functions
def load_yolo_model(size: str = "tiny"):
    """Load YOLO model with specified size."""
    cache_key = f"yolo_{size}"
    if cache_key in vision_models:
        return vision_models[cache_key]
        
    try:
        # Fix for the model name
        if size == "tiny":
            model_name = "yolov8n"  # Use nano model for tiny
        elif size == "medium":
            model_name = "yolov8m"
        elif size == "large":
            model_name = "yolov8l"
        else:
            model_name = f"yolov8{size}"
            
        from ultralytics import YOLO
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(model_name)
        model.to(device)
        
        vision_models[cache_key] = model
        print(f"✅ Loaded YOLO {size} model ({model_name}) on {device}")
        return model
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        return None

def load_scene_classification_model():
    """Load scene classification model."""
    cache_key = "scene_classification"
    if cache_key in vision_models:
        return vision_models[cache_key]
        
    try:
        import torch
        import timm
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use a lightweight model like MobileNetV3 pretrained on Places365 or ImageNet
        model = timm.create_model('mobilenetv3_large_100', pretrained=True)
        model.to(device)
        
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
        
        vision_models[cache_key] = (model, scene_categories)
        print(f"✅ Loaded scene classification model on {device}")
        return model, scene_categories
    except Exception as e:
        print(f"❌ Failed to load scene classification model: {e}")
        return None

def load_gdino_model():
    """Load Grounding DINO model for open-vocabulary detection."""
    cache_key = "gdino"
    if cache_key in vision_models:
        return vision_models[cache_key]
    
    try:
        # Import GDINO
        from groundingdino.util.inference import load_model, load_image, predict
        
        # Define config and checkpoint paths - adjust these to your installation
        config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        checkpoint_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
        
        # Load model
        model = load_model(config_path, checkpoint_path)
        
        vision_models[cache_key] = model
        print(f"✅ Loaded GroundingDINO model")
        return model
    except Exception as e:
        print(f"❌ Failed to load GroundingDINO model: {e}")
        return None

# Processing functions
def process_yolo(frames: List[np.ndarray], size: str = "tiny", confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """Run YOLO object detection on frames."""
    start_time = time.time()
    
    result = {
        "model_type": f"yolo_{size}",
        "confidence": 0.0,
        "objects": [],
        "processing_time": 0.0,
        "frames_processed": 0
    }
    
    try:
        # Load model
        model = load_yolo_model(size)
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
                    if conf >= confidence_threshold:
                        label = detections.names[int(class_id)]
                        result["objects"].append({
                            "label": label,
                            "confidence": float(conf),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        })
            except Exception as e:
                print(f"⚠️ Error processing frame with YOLO: {e}")
                continue
                    
            processed_count += 1
            
        # Group by label and take average confidence
        label_counts = {}
        for obj in result["objects"]:
            label = obj["label"]
            if label not in label_counts:
                label_counts[label] = {"count": 0, "conf_sum": 0}
            label_counts[label]["count"] += 1
            label_counts[label]["conf_sum"] += obj["confidence"]
        
        # Sort objects by count and confidence
        sorted_objects = []
        for label, info in sorted(
            label_counts.items(),
            key=lambda x: (x[1]["count"], x[1]["conf_sum"] / x[1]["count"]),
            reverse=True
        ):
            count = info["count"]
            avg_conf = info["conf_sum"] / count
            sorted_objects.append({
                "label": label,
                "confidence": avg_conf,
                "count": count
            })
        
        # Replace detailed objects with aggregated ones if there are many
        if len(result["objects"]) > 20:
            result["detailed_objects"] = result["objects"]
            result["objects"] = sorted_objects
        
        # Set overall confidence as average of top 3 detected objects
        if sorted_objects:
            top_objects = sorted_objects[:min(3, len(sorted_objects))]
            avg_conf = sum(obj["confidence"] for obj in top_objects) / len(top_objects)
            result["confidence"] = avg_conf
        
        # Complete result
        result["processing_time"] = time.time() - start_time
        result["frames_processed"] = processed_count
        
        return result
    except Exception as e:
        print(f"❌ Error running YOLO detection: {e}")
        result["processing_time"] = time.time() - start_time
        return result

def process_scene_classification(frames: List[np.ndarray], confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """Run scene classification on frames."""
    start_time = time.time()
    
    result = {
        "model_type": "scene_classification",
        "confidence": 0.0,
        "scene_categories": [],
        "processing_time": 0.0,
        "frames_processed": 0
    }
    
    try:
        # Load model
        model_data = load_scene_classification_model()
        if model_data is None:
            return result
            
        model, categories = model_data
        
        # Setup preprocessing
        import torchvision.transforms as transforms
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        from PIL import Image
        device = next(model.parameters()).device
        
        processed_count = 0
        category_scores = {category: 0.0 for category in categories}
        
        # Process frames
        for frame in frames:
            try:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(rgb_frame)
                
                # Preprocess image
                input_tensor = preprocess(pil_image).unsqueeze(0).to(device)
                
                # Run model
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
                # Update scores
                for i, category in enumerate(categories):
                    if i < len(probabilities):
                        category_scores[category] += float(probabilities[i])
            except Exception as e:
                print(f"⚠️ Error processing frame with scene classifier: {e}")
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
            if score >= confidence_threshold:
                result["scene_categories"].append({
                    "category": category,
                    "confidence": score
                })
        
        # Set overall confidence
        if result["scene_categories"]:
            result["confidence"] = sum(cat["confidence"] for cat in result["scene_categories"][:3]) / min(3, len(result["scene_categories"]))
        
        # Complete result
        result["processing_time"] = time.time() - start_time
        result["frames_processed"] = processed_count
        
        return result
    except Exception as e:
        print(f"❌ Error running scene classification: {e}")
        result["processing_time"] = time.time() - start_time
        return result

def process_gdino(frames: List[np.ndarray], text_prompts: List[str], confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """Run GroundingDINO open-vocabulary detection on frames."""
    start_time = time.time()
    
    result = {
        "model_type": "gdino",
        "confidence": 0.0,
        "objects": [],
        "processing_time": 0.0,
        "frames_processed": 0
    }
    
    try:
        # Load model
        model = load_gdino_model()
        if model is None:
            return result
            
        # Import prediction function
        from groundingdino.util.inference import predict
        
        # Process frames
        processed_count = 0
        for frame in frames:
            try:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create text prompt
                text_prompt = ".".join(text_prompts)
                
                # Run prediction
                boxes, logits, phrases = predict(
                    model=model,
                    image=rgb_frame,
                    caption=text_prompt,
                    box_threshold=confidence_threshold,
                    text_threshold=confidence_threshold
                )
                
                # Process results
                for box, logit, phrase in zip(boxes, logits, phrases):
                    x0, y0, x1, y1 = box.tolist()
                    result["objects"].append({
                        "label": phrase,
                        "confidence": float(logit),
                        "bbox": [int(x0), int(y0), int(x1), int(y1)]
                    })
            except Exception as e:
                print(f"⚠️ Error processing frame with GDINO: {e}")
                continue
                
            processed_count += 1
        
        # Set overall confidence
        if result["objects"]:
            result["confidence"] = sum(obj["confidence"] for obj in result["objects"]) / len(result["objects"])
        
        # Complete result
        result["processing_time"] = time.time() - start_time
        result["frames_processed"] = processed_count
        
        return result
    except Exception as e:
        print(f"❌ Error running GDINO detection: {e}")
        result["processing_time"] = time.time() - start_time
        return result

# Helper functions
async def download_frame(session, url: str) -> np.ndarray:
    """Download an image from URL and convert to OpenCV format."""
    try:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"Error downloading frame: HTTP {response.status}")
                return None
                
            data = await response.read()
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"❌ Error downloading frame from {url}: {e}")
        return None

async def download_frames(urls: List[str]) -> List[np.ndarray]:
    """Download multiple frames in parallel."""
    frames = []
    async with aiohttp.ClientSession() as session:
        tasks = [download_frame(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        frames = [frame for frame in results if frame is not None]
    return frames

# API endpoints
@app.post("/infer", response_model=VisionResponse)
async def infer(task_spec: TaskSpec):
    """Process frames with specified vision models."""
    start_time = time.time()
    
    # Validate model types
    for model_type in task_spec.model_types:
        if model_type not in ModelType.all_models():
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid model type: {model_type}"}
            )
    
    # Download frames
    frames = await download_frames(task_spec.frame_urls)
    
    if not frames:
        return JSONResponse(
            status_code=400,
            content={"error": "No valid frames found"}
        )
    
    # Process frames with each model
    results = []
    for model_type in task_spec.model_types:
        try:
            if model_type == ModelType.YOLO_TINY:
                result = process_yolo(frames, "tiny", task_spec.confidence)
            elif model_type == ModelType.YOLO_MEDIUM:
                result = process_yolo(frames, "medium", task_spec.confidence)
            elif model_type == ModelType.YOLO_LARGE:
                result = process_yolo(frames, "large", task_spec.confidence)
            elif model_type == ModelType.SCENE_CLASSIFICATION:
                result = process_scene_classification(frames, task_spec.confidence)
            elif model_type == ModelType.GDINO:
                # For GDINO, we need text prompts - use common objects as default
                result = process_gdino(
                    frames, 
                    ["person", "food", "drink", "building", "vehicle", "furniture"],
                    task_spec.confidence
                )
            else:
                continue
                
            results.append(result)
        except Exception as e:
            print(f"❌ Error processing {model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # Prepare response
    processing_time = time.time() - start_time
    response = {
        "results": results,
        "processing_time": processing_time,
        "frames_processed": len(frames),
        "models_used": task_spec.model_types
    }
    
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "models_loaded": list(vision_models.keys())}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
