#!/usr/bin/env python3
"""Training module for improving vector database template matching."""

import os
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

def train_from_video(url: str, force_add_to_vectors: bool = True, min_confidence: float = 0.3):
    """
    Process a video through the extraction pipeline and use it to train the vector database,
    even if similarity scores are below threshold.
    
    Args:
        url: URL of the video (Reel, TikTok, etc.)
        force_add_to_vectors: Whether to add to vector DB even if below threshold
        min_confidence: Minimum confidence to consider for template suggestions
        
    Returns:
        Dictionary with training results and template suggestions
    """
    # Import agent's processing function - directly use your existing code
    from agent import run, run_parallel_tasks, fetch_clip
    import tempfile
    import pathlib
    
    # Process video through normal pipeline - this leverages your existing agent code
    print(f"🎬 Processing video from: {url}")
    extraction_results = run(url)
    
    if "error" in extraction_results:
        return {"error": extraction_results["error"]}
    
    # Access the semantic prompt system - use your existing instance
    semantic_system = extraction_results.get("semantic_system")
    if not semantic_system:
        # If not included in results, create one (same as your agent.py)
        from semantic_prompt_system import SemanticPromptSystem
        semantic_system = SemanticPromptSystem(
            templates_dir="templates",
            modules_dir="modules",
            vectors_dir="vectors",
            cache_dir="cache"
        )
    
    # Get the fused text from the extraction results
    fused_text = "\n".join(
        filter(
            None,
            [
                extraction_results.get("speech_text", ""),
                extraction_results.get("frame_text", ""),
                extraction_results.get("caption_text", "")
            ]
        )
    )
    
    # Get template selection data but capture the raw selection details
    # Here we directly access the selector to get full details (same as in your semantic_prompt_system.py)
    template_info = semantic_system.template_selector.select_template(fused_text)
    
    # Even if below threshold, update the vector database
    if force_add_to_vectors:
        if template_info['method'] in ['vector', 'llm', 'llm_suggested_template']:
            # Get the template ID
            template_id = template_info.get('category')
            
            # Use the same update method your system already uses
            print(f"✨ Adding to vector database with template: {template_id}")
            semantic_system.template_selector.update_vector_database_with_successful_match(
                fused_text,
                template_id,
                quality_score=0.9  # Force high quality score for training purposes
            )
    
    # Check if we couldn't find a good template match
    needs_new_template = False
    template_suggestions = []
    
    if template_info['method'] in ['general', 'llm_composed']:
        needs_new_template = True
        
        # Use GPT to suggest potential template types
        template_suggestions = suggest_template_types(fused_text)
    
    # Return comprehensive training information - includes your agent's extraction results
    return {
        "url": url,
        "processed_data": extraction_results,
        "template_selection": {
            "method": template_info['method'],
            "category": template_info.get('category'),
            "score": template_info.get('score', 0),
            "confidence": template_info.get('confidence', 0),
            "similarity_scores": template_info.get('similarity_scores', {})
        },
        "needs_new_template": needs_new_template,
        "template_suggestions": template_suggestions,
        "vector_db_updated": force_add_to_vectors
    }


def suggest_template_types(content: str) -> list:
    """
    Use GPT to suggest potential template types for this content.
    
    Args:
        content: The extracted content from the video
        
    Returns:
        List of template suggestions with descriptions and examples
    """
    import openai
    import json
    
    # Create a prompt that asks GPT to suggest templates
    prompt = f"""Analyze this short-form video content and suggest specialized templates that would extract information well.
    
Content:
{content[:3000]}

Return JSON with suggested templates:
[
  {{
    "template_name": "template_id_to_use",
    "description": "Description of what this template would extract",
    "example_structure": {{
      // Example JSON structure this template would generate
    }}
  }}
]

Focus on the specific domain/category (food, travel, product review, etc.) and suggest 1-3 templates.
"""

    # Call the API
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "You suggest specialized templates for video content analysis."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "suggestions" in result:
            return result["suggestions"]
        if isinstance(result, dict) and "templates" in result:
            return result["templates"]
        return [result]
    except json.JSONDecodeError:
        # Handle error in JSON parsing
        return [{"template_name": "parsing_error", "description": "Could not parse suggestion response"}]


def batch_train_from_urls(url_list: list, output_path: str = "training_results.json"):
    """
    Process a batch of videos to train the vector database.
    
    Args:
        url_list: List of video URLs
        output_path: Path to save results
        
    Returns:
        Training results dictionary
    """
    import json
    import time
    from tqdm import tqdm
    
    results = {
        "processed": [],
        "errors": [],
        "template_needs": {},
        "timestamp": time.time()
    }
    
    # Process each URL
    for url in tqdm(url_list, desc="Training from videos"):
        try:
            result = train_from_video(url)
            
            if "error" in result:
                results["errors"].append({"url": url, "error": result["error"]})
            else:
                results["processed"].append(result)
                
                # Track needed templates
                if result["needs_new_template"] and result["template_suggestions"]:
                    for suggestion in result["template_suggestions"]:
                        template_name = suggestion["template_name"]
                        if template_name not in results["template_needs"]:
                            results["template_needs"][template_name] = {
                                "count": 0,
                                "description": suggestion["description"],
                                "example_structure": suggestion.get("example_structure", {})
                            }
                        results["template_needs"][template_name]["count"] += 1
                        
        except Exception as e:
            import traceback
            error_info = {"url": url, "error": str(e), "traceback": traceback.format_exc()}
            results["errors"].append(error_info)
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Processed {len(results['processed'])} videos with {len(results['errors'])} errors")
    print(f"📝 Identified {len(results['template_needs'])} potential new templates")
    
    # Print template needs summary
    if results["template_needs"]:
        print("\n🔍 Template Needs Summary:")
        for template_name, info in sorted(results["template_needs"].items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"  - {template_name}: {info['count']} videos, {info['description']}")
    
    return results


def generate_template_file(template_name: str, description: str, example_structure: Optional[Dict] = None):
    """
    Generate a template file based on the suggestion.
    
    Args:
        template_name: Name of the template
        description: Description of what the template extracts
        example_structure: Example JSON structure for the template
        
    Returns:
        Template data dictionary
    """
    # Create default structure if none provided
    if not example_structure:
        example_structure = {
            "content_type": template_name,
            "activities": [
                {
                    "place_name": "Example Place",
                    "genre": "Example Genre",
                    "availability": {
                        "city": "City",
                        "state": "State",
                        "country": "Country"
                    }
                }
            ]
        }
    
    # Generate template data
    template_data = {
        "system_prompt": f"""Extract information from short-form video content related to {template_name}.

{description}

Analyze the content carefully and extract structured information that includes all relevant details.

Return the information in the format specified below.""",
        "category": template_name,
        "subtypes": {},
        "output_format": example_structure
    }
    
    return template_data


def export_template_suggestions(results_file: str, output_dir: str, min_count: int = 2):
    """
    Export template suggestions from batch results as template files.
    
    Args:
        results_file: Path to batch results JSON file
        output_dir: Directory to save template files
        min_count: Minimum number of videos needing a template to export it
        
    Returns:
        List of exported template names
    """
    # Load batch results
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    template_needs = results.get("template_needs", {})
    exported = []
    
    for template_name, info in template_needs.items():
        if info["count"] >= min_count:
            # Create template file
            template_data = generate_template_file(
                template_name,
                info["description"],
                info.get("example_structure")
            )
            
            # Save template file
            file_path = os.path.join(output_dir, f"{template_name}.json")
            with open(file_path, "w") as f:
                json.dump(template_data, f, indent=2)
            
            exported.append(template_name)
    
    return exported