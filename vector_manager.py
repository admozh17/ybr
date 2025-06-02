# vector_manager.py
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class VectorManager:
    def __init__(
        self, model_name="BAAI/bge-small-en-v1.5", collection_name="brick_content"
    ):
        """
        Initialize the vector manager with a sentence transformer model.

        Args:
            model_name: Name of the SentenceTransformer model to use
            collection_name: Name of the collection in ChromaDB
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_size = self.model.get_sentence_embedding_dimension()
        self.collection_name = collection_name

        # Initialize ChromaDB client
        self.client = chromadb.Client(
            Settings(persist_directory="chroma_db", anonymized_telemetry=False)
        )

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

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
            for feedback_type in [
                "service_feedback",
                "food_feedback",
                "vibes_feedback",
                "miscellaneous_feedback",
            ]:
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

    def add_result(
        self, result_id: str, result_data: Dict[str, Any], url: str = None
    ) -> str:
        """
        Convert a result to vector embedding and add to database.

        Args:
            result_id: ID of the result (usually from database)
            result_data: Extracted data from a video
            url: URL of the original video

        Returns:
            The ID of the added vector
        """
        # Generate text for embedding
        text = self._generate_text_for_embedding(result_data)

        # Create embedding
        vector = self.model.encode(text).tolist()

        # Prepare metadata
        metadata = {
            "url": url,
            "content_type": result_data.get("content_type", "Unknown"),
            "result_id": result_id,
        }

        # Add activities metadata
        activities = result_data.get("activities", [])
        if activities:
            activity = activities[0]
            metadata["place_name"] = activity.get("place_name", "")
            metadata["genre"] = activity.get("genre", "")

            availability = activity.get("availability", {})
            metadata["city"] = availability.get("city", "")
            metadata["country"] = availability.get("country", "")
            metadata["region"] = availability.get("region", "")

        # Add to ChromaDB
        self.collection.add(
            ids=[str(result_id)],
            embeddings=[vector],
            metadatas=[metadata],
            documents=[text],
        )

        return result_id

    def search(
        self, query: str, limit: int = 10, filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
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

        # Prepare where filter if provided
        where = None
        if filters:
            where = {}
            for key, value in filters.items():
                if isinstance(value, list):
                    where[key] = {"$in": value}
                elif value:
                    where[key] = value

        # Perform search
        results = self.collection.query(
            query_embeddings=[query_vector], n_results=limit, where=where
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append(
                {
                    "id": results["ids"][0][i],
                    "score": results["distances"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
            )

        return formatted_results

    def delete_result(self, result_id: str):
        """Remove a result from the vector database."""
        self.collection.delete(ids=[str(result_id)])

    def update_result(
        self, result_id: str, result_data: Dict[str, Any], url: str = None
    ):
        """Update an existing result with new data."""
        # Delete old result and add new one
        self.delete_result(result_id)
        self.add_result(result_id, result_data, url)

    def get_similar_content(
        self, result_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find content similar to a specific result by ID.

        Args:
            result_id: ID of the result to find similar content for
            limit: Maximum number of results to return

        Returns:
            List of similar content with similarity scores
        """
        try:
            # Get the vector for the specified result
            result = self.collection.get(ids=[str(result_id)])
            if not result["embeddings"]:
                return []

            # Search for similar vectors
            results = self.collection.query(
                query_embeddings=[result["embeddings"][0]],
                n_results=limit + 1,  # Add 1 to exclude the result itself
            )

            # Format results (exclude the result itself)
            formatted_results = []
            for i in range(len(results["ids"][0])):
                if results["ids"][0][i] != str(result_id):  # Skip the result itself
                    formatted_results.append(
                        {
                            "id": results["ids"][0][i],
                            "score": results["distances"][0][i],
                            "metadata": results["metadatas"][0][i],
                        }
                    )

            return formatted_results[:limit]

        except Exception as e:
            print(f"Error getting similar content: {e}")
            return []
