#!/usr/bin/env python3
"""
Re-index data using fine-tuned model.
"""

import sys
import time
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from vector_manager import VectorManager

# Initialize Flask app
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///results.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Define Result model
class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), nullable=False)
    data = db.Column(db.JSON, nullable=False)
    timestamp = db.Column(db.DateTime)

def reindex_with_finetuned_model():
    """Re-index all results with the fine-tuned model."""
    # Initialize vector manager with fine-tuned model
    vector_manager = VectorManager(
        model_name="./brick-by-brick-embeddings",
        collection_name="brick_content_finetuned"  # New collection
    )
    
    with app.app_context():
        # Get all results
        results = Result.query.all()
        print(f"Found {len(results)} results to index")
        
        success = 0
        errors = 0
        start_time = time.time()
        
        # Re-index each result
        for i, result in enumerate(results):
            try:
                result_id = str(result.id)
                vector_manager.add_result(result_id, result.data, result.url)
                success += 1
                print(f"✅ Indexed {i+1}/{len(results)}: Result {result_id}")
            except Exception as e:
                errors += 1
                print(f"❌ Error indexing result {result.id}: {str(e)}")
        
        end_time = time.time()
        print(f"\nIndexing completed in {end_time - start_time:.2f} seconds")
        print(f"Total: {len(results)}, Success: {success}, Errors: {errors}")
    
    print("\nNext steps:")
    print("1. Update your web_app.py to use the new collection:")
    print("   vector_manager = VectorManager(model_name='./brick-by-brick-embeddings', collection_name='brick_content_finetuned')")
    print("2. Restart your application to start using the fine-tuned model")

if __name__ == "__main__":
    # Check if Qdrant is running
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        # Simple check to see if we can connect
        client.get_collections()
        print("✅ Connected to Qdrant successfully")
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {str(e)}")
        print("Make sure Qdrant is running. You can start it with Docker:")
        print("docker-compose up -d")
        sys.exit(1)
    
    # Confirm with user before re-indexing
    print("This will create a new collection 'brick_content_finetuned' with embeddings from the fine-tuned model.")
    print("Your existing collection will remain untouched.")
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        sys.exit(0)
    
    reindex_with_finetuned_model()