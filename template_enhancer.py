#!/usr/bin/env python3
"""Utility to enhance vector database with synthetic examples for improved template matching."""

import os
import json
import argparse
from pathlib import Path
import time
import openai
from vector_embeddings import VectorEmbeddingManager

def generate_synthetic_examples(template_file: str, num_examples: int = 5):
    """
    Generate synthetic examples for a template to enhance vector database matching.
    
    Args:
        template_file: Path to template JSON file
        num_examples: Number of synthetic examples to generate
        
    Returns:
        List of synthetic examples as text
    """
    # Load the template
    with open(template_file, "r") as f:
        template_data = json.load(f)
    
    template_id = Path(template_file).stem
    system_prompt = template_data.get("system_prompt", "")
    output_format = template_data.get("output_format", {})
    
    # Create a prompt for generating synthetic examples
    prompt = f"""Generate {num_examples} different synthetic examples of short-form video content that would match this template:

Template Name: {template_id}
Template Description: {system_prompt[:500]}

This template is designed to extract information in this format:
{json.dumps(output_format, indent=2) if output_format else "Standard JSON format"}

For each example, generate realistic text that represents what might be extracted from a video's:
1. Speech transcription
2. OCR text from the video
3. Video caption

Make each example different but clearly matching the template's purpose. Format as:

EXAMPLE 1:
SPEECH: [speech transcription]
OCR TEXT: [text visible in video]
CAPTION: [video caption]

EXAMPLE 2:
...

Each example should be 150-300 words total and contain realistic content that someone would actually post in a social media video.
"""

    # Call the API
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.8,  # Higher temperature for more variety
        messages=[
            {"role": "system", "content": "You generate synthetic examples of video content for training."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Process the response text to extract examples
    examples_text = response.choices[0].message.content
    examples = []
    
    # Split by "EXAMPLE" headers
    example_chunks = examples_text.split("EXAMPLE ")
    for chunk in example_chunks[1:]:  # Skip the first split which is empty or intro text
        # Clean up the example
        clean_chunk = chunk.strip()
        # Remove the example number if present
        if clean_chunk.startswith(tuple("0123456789")) and ":" in clean_chunk[:5]:
            clean_chunk = clean_chunk[clean_chunk.find(":")+1:].strip()
        
        examples.append(clean_chunk)
    
    return examples

def enhance_template_vectors(vectors_dir: str, templates_dir: str, template_id: str = None, 
                             num_examples: int = 5, force: bool = False):
    """
    Enhance vector database with synthetic examples for improved template matching.
    Uses the same vector mechanisms as the main agent system.
    
    Args:
        vectors_dir: Directory containing vector database
        templates_dir: Directory containing template files
        template_id: Specific template to enhance (None for all)
        num_examples: Number of synthetic examples per template
        force: Whether to regenerate for templates that already have synthetic examples
        
    Returns:
        Number of enhanced templates
    """
    try:
        # Use the same semantic prompt system your agent uses
        from semantic_prompt_system import SemanticPromptSystem
        
        # Create a semantic system with the same parameters as your agent
        semantic_system = SemanticPromptSystem(
            templates_dir=templates_dir,
            modules_dir="modules",
            vectors_dir=vectors_dir,
            cache_dir="cache"
        )
        
        # Access the vector manager directly from semantic system - same as your agent
        vector_manager = semantic_system.template_selector.vector_manager
        
        print(f"✅ Loaded existing vector database with {len(vector_manager.template_ids)} entries")
    except Exception as e:
        # Fallback to direct VectorEmbeddingManager if semantic system fails
        print(f"⚠️ Couldn't load semantic system: {e}. Falling back to direct vector manager.")
        vector_manager = VectorEmbeddingManager()
        
        # Try to load existing vector database
        if os.path.exists(os.path.join(vectors_dir, "template_vectors.faiss")):
            vector_manager.load(vectors_dir)
            print(f"✅ Loaded existing vector database with {len(vector_manager.template_ids)} entries")
    
    # Get template files to process
    template_files = []
    if template_id:
        template_path = os.path.join(templates_dir, f"{template_id}.json")
        if os.path.exists(template_path):
            template_files = [template_path]
        else:
            print(f"❌ Template file {template_path} not found")
            return 0
    else:
        template_paths = Path(templates_dir).glob("*.json")
        template_files = [str(path) for path in template_paths]
    
    # Track enhanced templates
    enhanced_count = 0
    templates_with_synthetic = set()
    
    # Check which templates already have synthetic examples
    for template_id in vector_manager.template_ids:
        if template_id.startswith("synthetic_"):
            base_template = template_id.split("_")[1]
            templates_with_synthetic.add(base_template)
    
    # Process each template
    for template_file in template_files:
        template_id = Path(template_file).stem
        
        # Skip if already has synthetic examples and not forced
        if template_id in templates_with_synthetic and not force:
            print(f"📝 Template {template_id} already has synthetic examples (use --force to regenerate)")
            continue
        
        print(f"🔄 Enhancing vector database for template: {template_id}")
        
        # Generate synthetic examples
        examples = generate_synthetic_examples(template_file, num_examples)
        print(f"✅ Generated {len(examples)} synthetic examples")
        
        # Load the template
        with open(template_file, "r") as f:
            template_data = json.load(f)
        
        # Add synthetic examples to vector database
        for i, example in enumerate(examples):
            synthetic_id = f"synthetic_{template_id}_{i+1}"
            
            # Add to vector database with same method as your agent
            vector_manager.add_template(
                synthetic_id,
                template_data,
                f"SYNTHETIC EXAMPLE for {template_id}:\n{example[:500]}"
            )
            
            print(f"  - Added {synthetic_id} to vector database")
        
        enhanced_count += 1
    
    # Save the enhanced vector database
    vector_manager.save(vectors_dir)
    print(f"💾 Saved enhanced vector database to {vectors_dir}")
    
    return enhanced_count

def analyze_template_coverage(vectors_dir: str):
    """
    Analyze template coverage in the vector database.
    
    Args:
        vectors_dir: Directory containing vector database
        
    Returns:
        Dictionary with coverage statistics
    """
    # Initialize vector manager
    vector_manager = VectorEmbeddingManager()
    
    # Try to load vector database
    if not os.path.exists(os.path.join(vectors_dir, "template_vectors.faiss")):
        print(f"❌ Vector database not found in {vectors_dir}")
        return {}
    
    vector_manager.load(vectors_dir)
    
    # Analyze template coverage
    template_counts = {}
    synthetic_counts = {}
    total_entries = len(vector_manager.template_ids)
    synthetic_entries = 0
    real_examples = 0
    
    for template_id in vector_manager.template_ids:
        parts = template_id.split("_")
        
        if template_id.startswith("synthetic_"):
            # Synthetic example
            base_template = parts[1]
            if base_template not in synthetic_counts:
                synthetic_counts[base_template] = 0
            synthetic_counts[base_template] += 1
            synthetic_entries += 1
        elif template_id.startswith("success_"):
            # Successful match example
            base_template = parts[1]
            if base_template not in template_counts:
                template_counts[base_template] = 0
            template_counts[base_template] += 1
            real_examples += 1
        elif template_id.startswith("example_"):
            # Example
            base_template = parts[1]
            if base_template not in template_counts:
                template_counts[base_template] = 0
            template_counts[base_template] += 1
            real_examples += 1
        else:
            # Base template or subtype
            if "_" in template_id:
                # Subtype
                base_template = parts[0]
            else:
                # Base template
                base_template = template_id
                
            if base_template not in template_counts:
                template_counts[base_template] = 0
            template_counts[base_template] += 1
    
    # Calculate coverage
    templates_without_synthetic = set(template_counts.keys()) - set(synthetic_counts.keys())
    
    # Return statistics
    return {
        "total_entries": total_entries,
        "synthetic_entries": synthetic_entries,
        "real_examples": real_examples,
        "total_templates": len(template_counts),
        "templates_with_synthetic": len(synthetic_counts),
        "templates_without_synthetic": list(templates_without_synthetic),
        "synthetic_counts": synthetic_counts,
        "template_counts": template_counts
    }

def main():
    parser = argparse.ArgumentParser(description="Enhance vector database with synthetic examples")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Enhance command
    enhance_parser = subparsers.add_parser("enhance", help="Enhance vector database with synthetic examples")
    enhance_parser.add_argument("--templates-dir", default="templates", help="Directory containing template files")
    enhance_parser.add_argument("--vectors-dir", default="vectors", help="Directory containing vector database")
    enhance_parser.add_argument("--template", help="Specific template ID to enhance (None for all)")
    enhance_parser.add_argument("--examples", type=int, default=5, help="Number of synthetic examples per template")
    enhance_parser.add_argument("--force", action="store_true", help="Force regeneration of synthetic examples")
    
    # Coverage command
    coverage_parser = subparsers.add_parser("coverage", help="Analyze template coverage in vector database")
    coverage_parser.add_argument("--vectors-dir", default="vectors", help="Directory containing vector database")
    coverage_parser.add_argument("--output", help="Output JSON file for coverage report")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "enhance":
        print(f"🔍 Enhancing vector database in {args.vectors_dir} with synthetic examples")
        
        enhanced = enhance_template_vectors(
            args.vectors_dir,
            args.templates_dir,
            args.template,
            args.examples,
            args.force
        )
        
        print(f"✅ Enhanced {enhanced} templates with synthetic examples")
        
    elif args.command == "coverage":
        print(f"🔍 Analyzing template coverage in vector database in {args.vectors_dir}")
        
        coverage = analyze_template_coverage(args.vectors_dir)
        
        if not coverage:
            return
            
        # Print coverage report
        print("\n📊 Vector Database Coverage Report:")
        print(f"  - Total entries: {coverage['total_entries']}")
        print(f"  - Real examples: {coverage['real_examples']}")
        print(f"  - Synthetic examples: {coverage['synthetic_entries']}")
        print(f"  - Total templates: {coverage['total_templates']}")
        print(f"  - Templates with synthetic examples: {coverage['templates_with_synthetic']}")
        
        # Show templates without synthetic examples
        if coverage['templates_without_synthetic']:
            print("\n⚠️ Templates without synthetic examples:")
            for template in sorted(coverage['templates_without_synthetic']):
                print(f"  - {template}")
                
        # Show template with most synthetic examples
        if coverage['synthetic_counts']:
            max_template = max(coverage['synthetic_counts'].items(), key=lambda x: x[1])
            print(f"\n🏆 Template with most synthetic examples: {max_template[0]} ({max_template[1]} examples)")
        
        # Save coverage report if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(coverage, f, indent=2)
            print(f"\n💾 Saved coverage report to {args.output}")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()