#!/usr/bin/env python3

import os
import sys
import time
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from vector_manager import VectorManager

# Initialize Flask app (for database access)
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

def index_all_results():
    """Index all existing results in the vector database."""
    print("Starting indexing of existing results...")
    
    # Initialize vector manager
    vector_manager = VectorManager()
    
    with app.app_context():
        # Get all results
        results = Result.query.all()
        total = len(results)
        print(f"Found {total} results to index")
        
        success = 0
        errors = 0
        
        for i, result in enumerate(results):
            try:
                # Add to vector database
                result_id = str(result.id)
                vector_manager.add_result(result_id, result.data, result.url)
                success += 1
                
                # Progress report
                print(f"✅ Indexed {i+1}/{total}: Result {result_id}")
            except Exception as e:
                errors += 1
                print(f"❌ Error indexing result {result.id}: {str(e)}")
        
        print(f"\nIndexing completed: {success} successes, {errors} errors")

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
        print("docker run -p 6333:6333 -v ./qdrant_data:/qdrant/storage qdrant/qdrant")
        sys.exit(1)
        
    start_time = time.time()
    index_all_results()
    end_time = time.time()
    print(f"Total indexing time: {end_time - start_time:.2f} seconds")