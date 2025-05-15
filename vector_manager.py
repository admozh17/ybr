# vector_manager.py
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

class VectorManager:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", collection_name="brick_content"):
        """
        Initialize the vector manager with a sentence transformer model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            collection_name: Name of the collection in Qdrant
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_size = self.model.get_sentence_embedding_dimension()
        self.collection_name = collection_name
        
        # Initialize Qdrant client (local server by default)
        self.client = QdrantClient(host="localhost", port=6333)
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create the vector collection if it doesn't exist."""
        collections = self.client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created new collection: {self.collection_name}")
    
    def _generate_text_for_embedding(self, result_data: Dict[str, Any]) -> str:
        """
        Generate a representative text from the structured data for embedding.
        
        Args:
            result_data: Extracted data from a video
            
        Returns:
            A text representation of the video content
        """
        parts = []
        
        # Add content type
        content_type = result_data.get("content_type", "Unknown")
        parts.append(f"Content Type: {content_type}")
        
        # Process each activity
        for idx, activity in enumerate(result_data.get("activities", [])):
            parts.append(f"Place: {activity.get('place_name', 'Unknown')}")
            parts.append(f"Genre: {activity.get('genre', 'Unknown')}")
            
            # Add cuisine if available
            if cuisine := activity.get("cuisine"):
                parts.append(f"Cuisine: {cuisine}")
            
            # Add location details
            availability = activity.get("availability", {})
            location_parts = []
            
            for field in ["street_address", "city", "state", "country", "region"]:
                if value := availability.get(field):
                    location_parts.append(value)
            
            if location_parts:
                parts.append(f"Location: {', '.join(location_parts)}")
            
            # Add dishes if available
            dishes = activity.get("dishes", {})
            if isinstance(dishes, dict):
                # Handle structured dishes
                for dish_type in ["explicitly_mentioned", "visually_shown"]:
                    for dish in dishes.get(dish_type, []):
                        if dish_name := dish.get("dish_name"):
                            parts.append(f"Dish: {dish_name}")
                            if feedback := dish.get("feedback"):
                                parts.append(f"  Feedback: {feedback}")
            elif isinstance(dishes, list):
                # Handle simple dish list
                for dish in dishes:
                    if isinstance(dish, dict) and (dish_name := dish.get("dish_name")):
                        parts.append(f"Dish: {dish_name}")
                        if feedback := dish.get("feedback"):
                            parts.append(f"  Feedback: {feedback}")
            
            # Add feedback if available
            feedback = activity.get("ratings_feedback", {})
            for feedback_type in ["service_feedback", "food_feedback", "vibes_feedback", "miscellaneous_feedback"]:
                if value := feedback.get(feedback_type):
                    parts.append(f"{feedback_type.replace('_', ' ').title()}: {value}")
        
        # Add raw text if available
        if speech_text := result_data.get("speech_text"):
            parts.append(f"Speech: {speech_text}")
        
        if frame_text := result_data.get("frame_text"):
            parts.append(f"OCR Text: {frame_text}")
        
        if caption_text := result_data.get("caption_text"):
            parts.append(f"Caption: {caption_text}")
        
        # Join all parts
        return "\n".join(parts)
    
    def _convert_id(self, id_value):
        """Convert ID to the correct format for Qdrant."""
        try:
            # Try to convert to int (preferred by Qdrant)
            return int(id_value)
        except (ValueError, TypeError):
            # If it can't be converted to int, return as string
            return str(id_value)
    
    def add_result(self, result_id: str, result_data: Dict[str, Any], url: str = None) -> str:
        """
        Convert a result to vector embedding and add to database.
        
        Args:
            result_id: ID of the result (usually from database)
            result_data: Extracted data from a video
            url: URL of the original video
            
        Returns:
            The ID of the added vector
        """
        # Convert ID to proper format
        qdrant_id = self._convert_id(result_id)
        
        # Generate text for embedding
        text = self._generate_text_for_embedding(result_data)
        
        # Create embedding
        vector = self.model.encode(text).tolist()
        
        # Prepare metadata for searching
        metadata = {
            "url": url,
            "content_type": result_data.get("content_type", "Unknown"),
            "result_id": result_id,  # Keep original ID in metadata
        }
        
        # Add activities metadata
        activities = result_data.get("activities", [])
        if activities:
            # Just store the first activity in metadata for filtering
            activity = activities[0]
            metadata["place_name"] = activity.get("place_name", "")
            metadata["genre"] = activity.get("genre", "")
            
            # Add location for geographical filtering
            availability = activity.get("availability", {})
            metadata["city"] = availability.get("city", "")
            metadata["country"] = availability.get("country", "")
            metadata["region"] = availability.get("region", "")
        
        # Add to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=qdrant_id,
                    vector=vector,
                    payload=metadata
                )
            ]
        )
        
        return result_id
    
    def search(self, query: str, limit: int = 10, 
               filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar content based on a text query.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Dict of filters to apply (e.g. {"genre": "restaurant"})
            
        Returns:
            List of search results with similarity scores and metadata
        """
        # Create embedding from query
        query_vector = self.model.encode(query).tolist()
        
        # Prepare filter conditions if provided
        filter_conditions = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    # Handle multiple values for a field (OR condition)
                    or_conditions = [
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=v)
                        ) for v in value if v
                    ]
                    if or_conditions:
                        conditions.append(models.Filter(
                            should=or_conditions
                        ))
                elif value:  # Ignore empty values
                    conditions.append(models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    ))
            
            if conditions:
                filter_conditions = models.Filter(
                    must=conditions
                )
        
        # Perform search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_conditions
        )
        
        # Format results
        results = []
        for hit in search_result:
            # Use the result_id from metadata (original format) 
            # or convert the Qdrant ID back to string if not in metadata
            original_id = hit.payload.get("result_id", hit.id)
            
            results.append({
                "id": original_id,
                "score": hit.score,
                "metadata": hit.payload
            })
        
        return results
    
    def delete_result(self, result_id: str):
        """Remove a result from the vector database."""
        # Convert ID to proper format
        qdrant_id = self._convert_id(result_id)
        
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[qdrant_id]
            )
        )
    
    def update_result(self, result_id: str, result_data: Dict[str, Any], url: str = None):
        """Update an existing result with new data."""
        # Simply overwrite
        self.add_result(result_id, result_data, url)
    
    def get_similar_content(self, result_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find content similar to a specific result by ID.
        
        Args:
            result_id: ID of the result to find similar content for
            limit: Maximum number of results to return
            
        Returns:
            List of similar content with similarity scores
        """
        # Convert ID to proper format
        qdrant_id = self._convert_id(result_id)
        
        try:
            # Retrieve the vector for the specified result
            vector = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[qdrant_id]
            )
            
            if not vector or not vector[0].vector:
                return []
            
            # Search for similar vectors
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector[0].vector,
                limit=limit + 1  # Add 1 to exclude the result itself
            )
            
            # Format results (exclude the result itself)
            results = []
            for hit in search_result:
                if hit.id != qdrant_id:  # Skip the result itself
                    # Use the result_id from metadata (original format)
                    original_id = hit.payload.get("result_id", hit.id)
                    
                    results.append({
                        "id": original_id,
                        "score": hit.score,
                        "metadata": hit.payload
                    })
            
            return results[:limit]
        
        except Exception as e:
            print(f"Error getting similar content: {e}")
            return []