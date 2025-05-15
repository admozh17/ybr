#!/usr/bin/env python3
"""Command-line tool for training video template matching system."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from template_trainer import train_from_video, batch_train_from_urls, export_template_suggestions

def main():
    parser = argparse.ArgumentParser(description="Train video template matching system")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single video training command - similar to your agent.py
    single_parser = subparsers.add_parser("train", help="Train from a single video")
    single_parser.add_argument("--url", required=True, help="URL of video to train from")
    single_parser.add_argument("--out", default="training_result.json", help="Output file for results")
    single_parser.add_argument("--no-force", action="store_true", help="Don't force add to vector DB if below threshold")
    single_parser.add_argument("--time", action="store_true", help="Show detailed timing information")
    
    # Batch training command
    batch_parser = subparsers.add_parser("batch", help="Train from a batch of videos")
    batch_parser.add_argument("--file", required=True, help="JSON file with list of URLs")
    batch_parser.add_argument("--out", default="batch_results.json", help="Output file for batch results")
    batch_parser.add_argument("--limit", type=int, help="Maximum number of videos to process")
    batch_parser.add_argument("--delay", type=int, default=0, help="Delay between videos in seconds")
    
    # Template suggestion export command
    export_parser = subparsers.add_parser("export-templates", help="Export template suggestions from batch results")
    export_parser.add_argument("--results", required=True, help="Batch results JSON file")
    export_parser.add_argument("--out-dir", default="template_suggestions", help="Output directory for template files")
    export_parser.add_argument("--min-count", type=int, default=2, help="Minimum video count to export a template")
    
    # Vector database stats command
    stats_parser = subparsers.add_parser("stats", help="Show vector database statistics")
    stats_parser.add_argument("--vectors-dir", default="vectors", help="Vector database directory")
    stats_parser.add_argument("--templates-dir", default="templates", help="Templates directory to check coverage")
    stats_parser.add_argument("--detailed", action="store_true", help="Show detailed template statistics")
    
    # Direct threshold override - useful for quick fixes
    threshold_parser = subparsers.add_parser("set-threshold", help="Set vector similarity threshold")
    threshold_parser.add_argument("--value", required=True, type=float, help="New threshold value (0.0-1.0)")
    threshold_parser.add_argument("--config", default="config.json", help="Config file path")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "train":
        try:
            start_time = time.time()
            print(f"🎓 Training from video: {args.url}")
            result = train_from_video(args.url, force_add_to_vectors=not args.no_force)
            
            # Save result
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
                
            print(f"✅ Saved training result to {args.out}")
            
            # Show brief summary
            print("\n📊 Training Summary:")
            template_selection = result.get('template_selection', {})
            print(f"  - Template method: {template_selection.get('method', 'unknown')}")
            print(f"  - Template category: {template_selection.get('category', 'unknown')}")
            
            # Show score or confidence based on method
            if template_selection.get('method') == 'vector':
                print(f"  - Vector score: {template_selection.get('score', 0):.4f}")
            else:
                print(f"  - Confidence: {template_selection.get('confidence', 0):.4f}")
                
            print(f"  - Vector DB updated: {result.get('vector_db_updated', False)}")
            print(f"  - Needs new template: {result.get('needs_new_template', False)}")
            
            if result.get('needs_new_template') and result.get('template_suggestions'):
                print("  - Suggested templates:")
                for suggestion in result.get('template_suggestions', []):
                    print(f"    * {suggestion.get('template_name')}: {suggestion.get('description')}")
            
            # Print timing information if requested
            if args.time:
                end_time = time.time()
                total_time = end_time - start_time
                
                # Get profile from processed data if available
                if "processed_data" in result and "performance_profile" in result["processed_data"]:
                    profile = result["processed_data"]["performance_profile"]
                    print("\n⏱️ Performance Profile:")
                    print(f"  📥 Download: {profile.get('download_time', 0):.2f}s")
                    print(f"  📝 Content Extraction: {profile.get('extraction_time', 0):.2f}s")
                    print(f"  🧠 LLM Processing: {profile.get('semantic_time', 0):.2f}s")
                    print(f"  🌍 Geocoding: {profile.get('geocode_time', 0):.2f}s")
                    
                print(f"  🏁 Total Training Time: {total_time:.2f}s")
                
        except Exception as e:
            print(f"💥 Critical error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Write error information to output file
            error_data = {"error": str(e), "url": args.url}
            with open(args.out, "w") as f:
                json.dump(error_data, f, indent=2)
            print(f"⚠️ Error information written to {args.out}")
        
    elif args.command == "batch":
        try:
            # Load URLs from file
            with open(args.file, "r") as f:
                data = json.load(f)
                
            # Handle different JSON formats
            url_list = []
            if isinstance(data, list):
                url_list = data
            elif isinstance(data, dict) and "urls" in data:
                url_list = data["urls"]
            else:
                print("❌ Error: Input file must contain a list of URLs or a dict with 'urls' key")
                sys.exit(1)
            
            # Apply limit if specified
            if args.limit and args.limit > 0:
                url_list = url_list[:args.limit]
                
            print(f"🎓 Training from {len(url_list)} videos")
            
            # Process with optional delay
            if args.delay > 0:
                print(f"ℹ️ Using {args.delay}s delay between videos to avoid rate limiting")
                
            results = batch_train_from_urls(url_list, output_path=args.out, delay_seconds=args.delay)
            
            print(f"✅ Saved batch results to {args.out}")
            
        except Exception as e:
            print(f"💥 Critical error: {str(e)}")
            import traceback
            traceback.print_exc()
            
    elif args.command == "export-templates":
        try:
            # Load batch results
            with open(args.results, "r") as f:
                results = json.load(f)
            
            # Create output directory
            os.makedirs(args.out_dir, exist_ok=True)
            
            template_needs = results.get("template_needs", {})
            exported = []
            
            for template_name, info in template_needs.items():
                if info["count"] >= args.min_count:
                    # Create template file
                    template_data = {
                        "system_prompt": f"Extract information from short-form video content related to {template_name}.\n\n{info['description']}\n\nExtract structured information matching the output format below.",
                        "category": template_name,
                        "subtypes": {},
                        "output_format": info.get("example_structure", {"content_type": template_name, "activities": []})
                    }
                    
                    # Save template file
                    file_path = os.path.join(args.out_dir, f"{template_name}.json")
                    with open(file_path, "w") as f:
                        json.dump(template_data, f, indent=2)
                    
                    exported.append(template_name)
            
            print(f"✅ Exported {len(exported)} template suggestions to {args.out_dir}")
            if exported:
                print("Templates exported:")
                for template in exported:
                    print(f"  - {template}")
                    
        except Exception as e:
            print(f"💥 Error exporting templates: {str(e)}")
            import traceback
            traceback.print_exc()
    
    elif args.command == "stats":
        # Show vector database statistics
        try:
            # Try to use your own system's SemanticPromptSystem first
            try:
                from semantic_prompt_system import SemanticPromptSystem
                
                semantic_system = SemanticPromptSystem(
                    templates_dir=args.templates_dir,
                    vectors_dir=args.vectors_dir
                )
                
                vector_manager = semantic_system.template_selector.vector_manager
                threshold = semantic_system.template_selector.vector_threshold
                
                print("\n📊 Vector Database Statistics:")
                print(f"  - Current similarity threshold: {threshold:.4f}")
                print(f"  - Templates in index: {len(vector_manager.template_ids)}")
                
            except Exception as e:
                # Fall back to direct vector manager
                print(f"⚠️ Couldn't load semantic system: {e}")
                from vector_embeddings import VectorEmbeddingManager
                
                vector_manager = VectorEmbeddingManager()
                vector_manager.load(args.vectors_dir)
                
                print("\n📊 Vector Database Statistics:")
                print(f"  - Templates in index: {len(vector_manager.template_ids)}")
            
            # Load available templates for coverage analysis
            available_templates = set()
            templates_path = Path(args.templates_dir)
            if templates_path.exists():
                for file_path in templates_path.glob("*.json"):
                    available_templates.add(file_path.stem)
            
            # Count by category
            categories = {}
            templates_in_vectors = set()
            synthetic_counts = {}
            real_example_counts = {}
            
            for template_id in vector_manager.template_ids:
                parts = template_id.split('_')
                
                if template_id.startswith("synthetic_"):
                    # Synthetic example
                    base_template = parts[1]
                    templates_in_vectors.add(base_template)
                    
                    if base_template not in synthetic_counts:
                        synthetic_counts[base_template] = 0
                    synthetic_counts[base_template] += 1
                    
                elif template_id.startswith("success_") or template_id.startswith("example_"):
                    # Real example
                    base_template = parts[1]
                    templates_in_vectors.add(base_template)
                    
                    if base_template not in real_example_counts:
                        real_example_counts[base_template] = 0
                    real_example_counts[base_template] += 1
                    
                else:
                    # Base template or subtype
                    if "_" in template_id:
                        # Subtype
                        base_template = parts[0]
                    else:
                        # Base template
                        base_template = template_id
                        
                    templates_in_vectors.add(base_template)
                
                # Count all entries by category
                category = parts[0]
                if category not in categories:
                    categories[category] = 0
                categories[category] += 1
            
            # Show template coverage information
            print(f"  - Available templates: {len(available_templates)}")
            print(f"  - Templates in vector DB: {len(templates_in_vectors)}")
            
            # Templates missing from vector DB
            missing_templates = available_templates - templates_in_vectors
            if missing_templates:
                print(f"\n⚠️ Templates missing from vector database ({len(missing_templates)}):")
                for template in sorted(missing_templates):
                    print(f"  - {template}")
            
            print("\n  Categories:")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                if args.detailed or count > 10:  # Show all categories if detailed, otherwise just larger ones
                    print(f"    * {category}: {count} entries")
            
            # Detailed template statistics
            if args.detailed:
                print("\n📈 Detailed Template Statistics:")
                
                # Sort templates by number of examples
                templates_by_examples = []
                for template in templates_in_vectors:
                    real_count = real_example_counts.get(template, 0)
                    synthetic_count = synthetic_counts.get(template, 0)
                    templates_by_examples.append((template, real_count, synthetic_count))
                
                # Sort by total examples (real + synthetic)
                templates_by_examples.sort(key=lambda x: x[1] + x[2], reverse=True)
                
                for template, real, synthetic in templates_by_examples:
                    print(f"  - {template}: {real} real examples, {synthetic} synthetic examples")
                
        except Exception as e:
            print(f"❌ Error analyzing vector database: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.command == "set-threshold":
        try:
            # Read config file if it exists
            config = {}
            if os.path.exists(args.config):
                with open(args.config, "r") as f:
                    config = json.load(f)
            
            # Update threshold
            old_threshold = config.get("vector_threshold", 0.6)
            config["vector_threshold"] = args.value
            
            # Write config file
            with open(args.config, "w") as f:
                json.dump(config, f, indent=2)
                
            print(f"✅ Updated vector threshold from {old_threshold:.2f} to {args.value:.2f}")
            print(f"ℹ️ Configuration saved to {args.config}")
            
            # Update threshold in semantic system if it's directly accessible
            try:
                from semantic_prompt_system import SemanticPromptSystem
                
                semantic_system = SemanticPromptSystem()
                if hasattr(semantic_system, 'template_selector'):
                    if hasattr(semantic_system.template_selector, 'vector_threshold'):
                        semantic_system.template_selector.vector_threshold = args.value
                        print("✅ Updated threshold in currently loaded semantic system")
            except Exception as e:
                print(f"⚠️ Note: Could not update loaded system: {e}")
                print("   The threshold will be applied when the system is next loaded.")
                
        except Exception as e:
            print(f"❌ Error updating threshold: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()