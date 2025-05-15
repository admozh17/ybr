#!/usr/bin/env python3
"""
Fine-tune embedding model for Brick by Brick using extracted data.
This script leverages all the structured data from previous extractions
to create high-quality training examples for domain-specific search.
"""

import os
import sys
import json
import random
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

# Initialize Flask app for database access
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///results.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Define Result model (matching the one in web_app.py)
class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), nullable=False)
    data = db.Column(db.JSON, nullable=False)
    timestamp = db.Column(db.DateTime)

# --- Data Generation Functions ---

def generate_rich_document(activity: Dict[str, Any]) -> str:
    """
    Generate a rich text representation of an activity for embedding.
    
    Args:
        activity: Activity data extracted from a video
        
    Returns:
        Text representation of the activity
    """
    document_parts = []
    
    # Basic info
    place_name = activity.get('place_name', '')
    document_parts.append(f"Place: {place_name}")
    
    if genre := activity.get('genre', ''):
        document_parts.append(f"Genre: {genre}")
    
    if cuisine := activity.get('cuisine', ''):
        document_parts.append(f"Cuisine: {cuisine}")
    
    if vibes := activity.get('vibes', ''):
        document_parts.append(f"Vibes: {vibes}")
    
    # Location details
    availability = activity.get('availability', {})
    
    for field in ["street_address", "city", "state", "country", "region"]:
        if value := availability.get(field):
            document_parts.append(f"{field.replace('_', ' ').title()}: {value}")
    
    # Dishes information
    dishes = activity.get('dishes', {})
    dish_list = []
    
    if isinstance(dishes, dict):
        # Handle structured dishes format
        for dish_type in ["explicitly_mentioned", "visually_shown"]:
            for dish in dishes.get(dish_type, []):
                if dish_name := dish.get('dish_name'):
                    dish_list.append(dish_name)
                    if feedback := dish.get('feedback'):
                        document_parts.append(f"Dish: {dish_name} - {feedback}")
                    else:
                        document_parts.append(f"Dish: {dish_name}")
    elif isinstance(dishes, list):
        # Handle flat dishes list
        for dish in dishes:
            if isinstance(dish, dict) and (dish_name := dish.get('dish_name')):
                dish_list.append(dish_name)
                if feedback := dish.get('feedback'):
                    document_parts.append(f"Dish: {dish_name} - {feedback}")
                else:
                    document_parts.append(f"Dish: {dish_name}")
    
    # Feedback information
    feedback = activity.get('ratings_feedback', {})
    for feedback_type in ["service_feedback", "food_feedback", "vibes_feedback", "miscellaneous_feedback"]:
        if value := feedback.get(feedback_type):
            document_parts.append(f"{feedback_type.replace('_', ' ').title()}: {value}")
    
    # Return the joined document text
    return "\n".join(document_parts)


def generate_queries_for_activity(activity: Dict[str, Any]) -> List[str]:
    """
    Generate diverse search queries for an activity.
    
    Args:
        activity: Activity data extracted from a video
        
    Returns:
        List of potential search queries
    """
    queries = []
    
    # Extract key fields
    place_name = activity.get('place_name', '')
    genre = activity.get('genre', '')
    cuisine = activity.get('cuisine', '')
    vibes = activity.get('vibes', '')
    
    # Get location info
    availability = activity.get('availability', {})
    city = availability.get('city', '')
    country = availability.get('country', '')
    region = availability.get('region', '')
    
    # Get dishes
    dishes = activity.get('dishes', {})
    dish_list = []
    
    if isinstance(dishes, dict):
        for dish_type in ["explicitly_mentioned", "visually_shown"]:
            for dish in dishes.get(dish_type, []):
                if dish_name := dish.get('dish_name'):
                    dish_list.append(dish_name)
    elif isinstance(dishes, list):
        for dish in dishes:
            if isinstance(dish, dict) and (dish_name := dish.get('dish_name')):
                dish_list.append(dish_name)
    
    # 1. Name-based queries
    if place_name:
        queries.append(place_name)
        queries.append(f"about {place_name}")
    
    # 2. Genre-based queries
    if genre:
        queries.append(genre)
        queries.append(f"{genre} places")
        if city:
            queries.append(f"{genre} in {city}")
        if country:
            queries.append(f"{genre} in {country}")
        if region:
            queries.append(f"{genre} in {region}")
    
    # 3. Cuisine-based queries
    if cuisine:
        queries.append(f"{cuisine} food")
        queries.append(f"{cuisine} cuisine")
        if city:
            queries.append(f"{cuisine} in {city}")
        if genre:
            queries.append(f"{cuisine} {genre}")
    
    # 4. Vibe-based queries
    if vibes:
        queries.append(f"{vibes} places")
        if genre:
            queries.append(f"{vibes} {genre}")
        if city:
            queries.append(f"{vibes} places in {city}")
    
    # 5. Location-based queries
    if city:
        queries.append(f"places in {city}")
        queries.append(f"where to eat in {city}")
        if genre:
            queries.append(f"{genre} in {city}")
        if cuisine:
            queries.append(f"{cuisine} food in {city}")
    
    if region:
        queries.append(f"places in {region}")
        if genre:
            queries.append(f"{genre} in {region}")
    
    # 6. Dish-based queries
    for dish in dish_list[:3]:  # Limit to 3 dishes to avoid too many queries
        queries.append(f"places with {dish}")
        if city:
            queries.append(f"{dish} in {city}")
    
    # Remove duplicates and return
    return list(set(queries))


def extract_keywords_from_text(text: str) -> List[str]:
    """
    Extract descriptive keywords from feedback text.
    
    Args:
        text: Feedback text
        
    Returns:
        List of descriptive keywords
    """
    if not text:
        return []
    
    # List of descriptive adjectives commonly used in reviews
    descriptive_words = [
        "delicious", "amazing", "excellent", "fantastic", "wonderful", "great", 
        "tasty", "flavorful", "cozy", "trendy", "upscale", "casual", "authentic",
        "unique", "friendly", "crowded", "quiet", "romantic", "traditional", "modern",
        "fresh", "crispy", "juicy", "sweet", "spicy", "savory", "creamy", "popular",
        "famous", "best", "signature", "specialty", "favorite", "local", "hidden"
    ]
    
    # Extract words that match our descriptive list
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [word for word in words if word in descriptive_words]
    
    # Return unique keywords or a subset of words if no keywords found
    return list(set(keywords)) if keywords else words[:3]


def extract_natural_queries(text: str, place_name: str) -> List[str]:
    """
    Extract natural-sounding queries from raw text.
    
    Args:
        text: Raw text extracted from video
        place_name: Name of the place to help validate relevance
        
    Returns:
        List of natural query phrases
    """
    queries = []
    
    # Extract phrases that might be natural queries
    query_patterns = [
        r"(?:where to find|where to get) (.{3,30})",
        r"(?:best|great|amazing) (.{3,30}) in (.{3,30})",
        r"(?:recommend|try|must try) (.{3,30})",
        r"(?:looking for|want to find) (.{3,30})",
        r"(?:how to|where can i) (.{3,30})"
    ]
    
    for pattern in query_patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            if isinstance(match, tuple):
                query = " ".join(match)
            else:
                query = match
                
            if 3 <= len(query) <= 50:  # Reasonable query length
                # Only add if it seems related to the place
                if place_name.lower() in text.lower():
                    queries.append(query)
    
    return list(set(queries))  # Remove duplicates


def generate_training_examples(results) -> List[Dict[str, Any]]:
    """
    Generate comprehensive training examples from all results.
    
    Args:
        results: List of Result objects from database
        
    Returns:
        List of training examples with query, document, and label
    """
    examples = []
    
    print("Generating positive training examples...")
    # Process each result
    for result in results:
        # Process each activity in the result
        for activity_idx, activity in enumerate(result.data.get('activities', [])):
            # Skip if missing essential data
            place_name = activity.get('place_name', '')
            if not place_name:
                continue
            
            # Generate document text
            document_text = generate_rich_document(activity)
            
            # Generate diverse queries
            queries = generate_queries_for_activity(activity)
            
            # Add feedback-based queries if available
            feedback = activity.get('ratings_feedback', {})
            for feedback_type in ["food_feedback", "vibes_feedback"]:
                if text := feedback.get(feedback_type):
                    keywords = extract_keywords_from_text(text)
                    for keyword in keywords[:2]:  # Limit to 2 keywords
                        if feedback_type == "food_feedback":
                            queries.append(f"{keyword} food")
                        else:
                            queries.append(f"{keyword} atmosphere")
            
            # Include raw text for additional context if available
            raw_text = result.data.get('speech_text', '') + ' ' + result.data.get('caption_text', '')
            if raw_text and len(raw_text) > 50:
                # Extract potential user queries from raw text
                natural_queries = extract_natural_queries(raw_text, place_name)
                queries.extend(natural_queries)
            
            # Create positive examples for each query
            for query in queries:
                examples.append({
                    "query": query,
                    "document": document_text,
                    "label": 1.0,  # Positive match
                    "place_name": place_name,
                    "genre": activity.get('genre', ''),
                    "city": activity.get('availability', {}).get('city', '')
                })
    
    print(f"Generated {len(examples)} positive examples")
    
    # Create hard negative examples
    print("Generating hard negative examples...")
    hard_negatives = generate_hard_negatives(examples)
    examples.extend(hard_negatives)
    
    print(f"Added {len(hard_negatives)} hard negative examples")
    
    # Create truly negative examples
    print("Generating random negative examples...")
    negative_examples = generate_random_negatives(examples)
    examples.extend(negative_examples)
    
    print(f"Added {len(negative_examples)} random negative examples")
    print(f"Total training examples: {len(examples)}")
    
    # Shuffle examples
    random.shuffle(examples)
    return examples


def generate_hard_negatives(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate hard negative examples - similar but not correct matches.
    
    Args:
        examples: List of positive examples
        
    Returns:
        List of hard negative examples
    """
    hard_negatives = []
    
    # Group activities by genre
    genre_groups = {}
    for example in examples:
        genre = example.get('genre', '')
        if genre and genre not in genre_groups:
            genre_groups[genre] = []
        
        if genre:
            genre_groups[genre].append(example)
    
    # Create hard negatives: same genre, different place
    positive_sample = examples[:min(300, len(examples))]
    for ex in positive_sample:
        genre = ex.get('genre', '')
        place_name = ex.get('place_name', '')
        
        if not genre or not place_name or genre not in genre_groups:
            continue
        
        # Find another place of the same genre
        same_genre = [p for p in genre_groups[genre] 
                     if p.get('place_name') != place_name]
        
        if same_genre:
            hard_neg = random.choice(same_genre)
            
            hard_negatives.append({
                "query": ex["query"],
                "document": hard_neg["document"],
                "label": 0.3,  # Partial match - same genre but wrong place
                "place_name": hard_neg.get('place_name', ''),
                "genre": genre
            })
    
    return hard_negatives


def generate_random_negatives(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate completely unrelated negative examples.
    
    Args:
        examples: List of existing examples
        
    Returns:
        List of negative examples
    """
    negatives = []
    documents = [ex["document"] for ex in examples]
    
    # Create negative examples
    for i, ex in enumerate(examples[:min(200, len(examples))]):
        if i % 5 != 0:  # Only use 20% of examples for negatives to keep balanced
            continue
            
        # Get a random unrelated document
        other_docs = [doc for doc in documents 
                     if ex.get('place_name', '') not in doc 
                     and ex.get('genre', '') not in doc]
        
        if other_docs:
            neg_doc = random.choice(other_docs)
            
            negatives.append({
                "query": ex["query"],
                "document": neg_doc,
                "label": 0.0,  # Complete non-match
            })
    
    return negatives


# --- Fine-Tuning Functions ---

def prepare_training_data(examples: List[Dict[str, Any]]) -> Tuple[List, List]:
    """
    Prepare training and validation datasets.
    
    Args:
        examples: List of training examples
        
    Returns:
        Tuple of (training InputExamples, validation InputExamples)
    """
    # Split into train and validation (90/10 split)
    val_size = max(10, int(len(examples) * 0.1))
    train_examples = examples[:-val_size]
    val_examples = examples[-val_size:]
    
    print(f"Training set: {len(train_examples)} examples")
    print(f"Validation set: {len(val_examples)} examples")
    
    # Convert to InputExample format
    train_inputs = [
        InputExample(texts=[ex["query"], ex["document"]], label=ex["label"])
        for ex in train_examples
    ]
    
    val_inputs = [
        InputExample(texts=[ex["query"], ex["document"]], label=ex["label"])
        for ex in val_examples
    ]
    
    return train_inputs, val_inputs


def fine_tune_model(model_name: str, train_inputs: List, val_inputs: List, output_path: str):
    """
    Fine-tune the embedding model.
    
    Args:
        model_name: Name of base model to fine-tune
        train_inputs: List of training InputExamples
        val_inputs: List of validation InputExamples
        output_path: Where to save the fine-tuned model
    """
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load base model
    print(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    
    # Create evaluator
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1=[ex.texts[0] for ex in val_inputs],
        sentences2=[ex.texts[1] for ex in val_inputs],
        labels=[ex.label for ex in val_inputs]
    )
    
    # Create data loader
    train_batch_size = 16  # Adjust based on available memory
    train_dataloader = DataLoader(train_inputs, shuffle=True, batch_size=train_batch_size)
    
    # Set up the loss
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Configure training parameters
    num_epochs = 3
    warmup_steps = int(len(train_dataloader) * 0.1)
    
    # Train the model
    print(f"Starting fine-tuning for {num_epochs} epochs")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        evaluation_steps=200,
        output_path=output_path,
        show_progress_bar=True
    )
    
    print(f"Model fine-tuning completed. Model saved to: {output_path}")


def evaluate_model(base_model_name: str, fine_tuned_path: str, test_examples: List[Dict[str, Any]]):
    """
    Evaluate and compare base model vs fine-tuned model.
    
    Args:
        base_model_name: Name of base model
        fine_tuned_path: Path to fine-tuned model
        test_examples: List of test examples
    """
    # Load models
    base_model = SentenceTransformer(base_model_name)
    fine_tuned_model = SentenceTransformer(fine_tuned_path)
    
    # Prepare test queries and documents
    test_queries = [ex["query"] for ex in test_examples[:20]]  # Sample 20 queries
    test_documents = list(set([ex["document"] for ex in test_examples]))[:100]  # Sample 100 documents
    
    print(f"Evaluating with {len(test_queries)} queries against {len(test_documents)} documents")
    
    # Compare base vs fine-tuned for each query
    from sentence_transformers import util
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Base model results
        query_embedding = base_model.encode(query, convert_to_tensor=True)
        doc_embeddings = base_model.encode(test_documents, convert_to_tensor=True)
        
        cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        top_results = torch.topk(cos_scores, k=3)
        
        print("Base model top results:")
        for score, idx in zip(top_results[0], top_results[1]):
            # Extract the place name from the document for display
            place_match = re.search(r"Place: (.*)", test_documents[idx])
            place_name = place_match.group(1) if place_match else "Unknown"
            print(f"  - {place_name} (Score: {score:.4f})")
        
        # Fine-tuned model results
        query_embedding = fine_tuned_model.encode(query, convert_to_tensor=True)
        doc_embeddings = fine_tuned_model.encode(test_documents, convert_to_tensor=True)
        
        cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        top_results = torch.topk(cos_scores, k=3)
        
        print("Fine-tuned model top results:")
        for score, idx in zip(top_results[0], top_results[1]):
            place_match = re.search(r"Place: (.*)", test_documents[idx])
            place_name = place_match.group(1) if place_match else "Unknown"
            print(f"  - {place_name} (Score: {score:.4f})")


# --- Main Function ---

def main():
    """Main function to run the fine-tuning process."""
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune embedding model for Brick by Brick")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="Base model to fine-tune")
    parser.add_argument("--output", default="./brick-by-brick-embeddings", help="Output directory for fine-tuned model")
    parser.add_argument("--examples-file", help="Save generated examples to JSON file")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model after fine-tuning")
    
    args = parser.parse_args()
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA not available. Training will be slower on CPU.")
        response = input("Continue with CPU training? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Create output directory if it doesn't exist
    os.makedirs(Path(args.output).parent, exist_ok=True)
    
    # Run fine-tuning process
    with app.app_context():
        # Get all results from database
        results = Result.query.all()
        print(f"Found {len(results)} results in database")
        
        if len(results) == 0:
            print("No results found in database. Cannot proceed with fine-tuning.")
            sys.exit(1)
        
        # Generate training examples
        examples = generate_training_examples(results)
        
        # Save examples to file if requested
        if args.examples_file:
            with open(args.examples_file, 'w') as f:
                json.dump(examples, f, indent=2)
            print(f"Saved {len(examples)} examples to {args.examples_file}")
        
        # Prepare training data
        train_inputs, val_inputs = prepare_training_data(examples)
        
        # Fine-tune the model
        fine_tune_model(args.model, train_inputs, val_inputs, args.output)
        
        # Evaluate if requested
        if args.evaluate:
            evaluate_model(args.model, args.output, examples)
        
        print("\nNext steps:")
        print(f"1. Update your vector_manager.py to use '{args.output}' as the model")
        print("2. Re-index your data with the new model")
        print("3. Enjoy improved search results!")


if __name__ == "__main__":
    main()