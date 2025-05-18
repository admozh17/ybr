#!/usr/bin/env python3
"""CLIP gating utility for filtering video frames based on content relevance."""

import torch
import numpy as np
import clip
from typing import List, Dict, Any, Union, Optional
from PIL import Image
import cv2

class ClipGate:
    """Uses CLIP model to filter frames based on text prompts."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the CLIP gating module.
        
        Args:
            device: Device to run CLIP on ('cuda' or 'cpu'). If None, auto-detects.
        """
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.preprocess = None
        self.loaded = False
    
    def load_model(self):
        """Load CLIP model if not already loaded."""
        if not self.loaded:
            try:
                # Load CLIP model
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                self.loaded = True
                print(f"✅ Loaded CLIP model on {self.device}")
            except Exception as e:
                print(f"❌ Error loading CLIP model: {e}")
                raise
    
    def prepare_image_batch(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Convert OpenCV frames to preprocessed CLIP input tensors.
        
        Args:
            frames: List of OpenCV BGR frames
            
        Returns:
            Batch tensor of preprocessed frames
        """
        # Convert BGR frames to RGB PIL images
        images = []
        for frame in frames:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            # Preprocess for CLIP
            processed = self.preprocess(pil_image)
            images.append(processed)
        
        # Stack images into a batch tensor
        if not images:
            return None
            
        image_batch = torch.stack(images).to(self.device)
        return image_batch
    
    def prepare_text_prompts(self, entities: List[str]) -> torch.Tensor:
        """
        Convert text entities to CLIP text embeddings.
        
        Args:
            entities: List of entity strings to search for
            
        Returns:
            Tensor of tokenized text prompts
        """
        if not entities:
            return None
            
        # Add descriptions to make entities more recognizable
        text_prompts = []
        for entity in entities:
            # Create variations of the entity prompt to improve detection
            text_prompts.extend([
                f"a photo of {entity}",
                f"{entity}",
                f"an image containing {entity}",
                f"a scene with {entity}"
            ])
        
        # Tokenize text
        tokenized = clip.tokenize(text_prompts).to(self.device)
        return tokenized
    
    def filter_frames(self, 
                     frames: List[np.ndarray], 
                     entities: List[str], 
                     threshold: float = 0.2,
                     max_frames: int = 10) -> Dict[str, Any]:
        """
        Filter frames using CLIP to find those matching the given entities.
        
        Args:
            frames: List of OpenCV BGR frames
            entities: List of text entities to detect
            threshold: Cosine similarity threshold for matching
            max_frames: Maximum number of frames to return
            
        Returns:
            Dictionary with filtered frames and match scores
        """
        if not frames or not entities:
            return {"kept_frame_indices": list(range(min(len(frames), max_frames))), 
                    "similarity_scores": {}}
        
        # Load model
        if not self.loaded:
            self.load_model()
        
        # Convert frames to CLIP format
        image_batch = self.prepare_image_batch(frames)
        
        # Process text prompts
        text_batch = self.prepare_text_prompts(entities)
        
        # Run CLIP comparison
        with torch.no_grad():
            # Encode images
            image_features = self.model.encode_image(image_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Encode text
            text_features = self.model.encode_text(text_batch)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            similarities = (100.0 * image_features @ text_features.T)
            
            # For each frame, take the max similarity across all text prompts
            max_similarities, _ = similarities.max(dim=1)
            
            # Convert to list
            similarity_scores = max_similarities.cpu().numpy().tolist()
        
        # Create frame index to entity mapping for matches
        matches = {}
        for i, score in enumerate(similarity_scores):
            matches[i] = score
        
        # Sort frames by similarity score
        sorted_frames = sorted(matches.items(), key=lambda x: x[1], reverse=True)
        
        # Filter frames above threshold, limited to max_frames
        kept_indices = []
        kept_scores = {}
        
        # Always keep at least one frame (the best match)
        if sorted_frames:
            kept_indices.append(sorted_frames[0][0])
            kept_scores[sorted_frames[0][0]] = sorted_frames[0][1]
        
        # Add additional frames that meet the threshold
        for idx, score in sorted_frames[1:]:
            if score >= threshold and len(kept_indices) < max_frames:
                kept_indices.append(idx)
                kept_scores[idx] = score
        
        # If we have too few frames, add more up to max_frames
        if len(kept_indices) < max_frames and len(kept_indices) < len(frames):
            for idx, score in sorted_frames:
                if idx not in kept_indices and len(kept_indices) < max_frames:
                    kept_indices.append(idx)
                    kept_scores[idx] = score
        
        # Sort indices for sequential processing
        kept_indices.sort()
        
        print(f"🔍 CLIP gate selected {len(kept_indices)}/{len(frames)} frames using {len(entities)} entities")
        print(f"   Threshold: {threshold}, Top score: {sorted_frames[0][1] if sorted_frames else 0.0}")
        
        return {
            "kept_frame_indices": kept_indices,
            "similarity_scores": kept_scores
        }


# Example usage
if __name__ == "__main__":
    import sys
    import pathlib
    import cv2
    
    def extract_frames(video_path, num_frames=10):
        """Extract equally spaced frames from video."""
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python clip_gate.py video_path entity1 entity2 ...")
        sys.exit(1)
    
    video_path = pathlib.Path(sys.argv[1])
    entities = sys.argv[2:]
    
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"🎬 Extracting frames from {video_path}...")
    frames = extract_frames(video_path)
    print(f"✅ Extracted {len(frames)} frames")
    
    print(f"🔍 Filtering frames for entities: {', '.join(entities)}")
    gate = ClipGate()
    result = gate.filter_frames(frames, entities)
    
    print(f"✅ Kept {len(result['kept_frame_indices'])} frames:")
    for idx in result['kept_frame_indices']:
        score = result['similarity_scores'].get(idx, 0.0)
        print(f"   Frame {idx}: Score {score:.2f}")