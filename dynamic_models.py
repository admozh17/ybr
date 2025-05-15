#!/usr/bin/env python3
"""
Dynamic Pydantic Model Generation for Template-Based Extraction
This module provides functionality to create dynamic Pydantic models based on template data,
making the validation system adaptable to changing templates.
"""

import json
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, create_model

# Field name standardization mappings
# IMPORTANT: These are only used for handling LLM output variations,
# not for transforming the fields in the template itself
FIELD_MAPPINGS = {
    # Place name variations
    "restaurant_name": "place_name",
    "location_name": "place_name",
    "venue_name": "place_name",
    "establishment": "place_name",
    "name": "place_name",
    
    # Genre/type variations
    "type": "genre",
    "category": "genre",
    "establishment_type": "genre",
    "business_type": "genre",
    
    # Address variations
    "address": "street_address",
    "street": "street_address"
}

# Create a base model with the configuration we need
class DynamicBaseModel(BaseModel):
    model_config = {
        'extra': 'allow',
        'populate_by_name': True
    }

def create_dynamic_models_from_template(template_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate Pydantic models dynamically based on template output format.
    
    Args:
        template_data: The template data containing output_format and classification_tree
        
    Returns:
        Dictionary of generated model classes keyed by name
    """
    output_format = template_data.get('output_format', {})
    template_id = template_data.get('category', 'generic')
    
    # Create model classes dictionary
    model_classes = {}
    
    # Create nested models from template
    if 'activities' in output_format and isinstance(output_format['activities'], list) and output_format['activities']:
        activity_schema = output_format['activities'][0]
        
        # Process each section that might be a nested model
        for key, value in activity_schema.items():
            if isinstance(value, dict):
                # Create fields for nested model
                nested_fields = {}
                for sub_key, sub_value in value.items():
                    # Make all fields optional with appropriate types
                    if isinstance(sub_value, list):
                        nested_fields[sub_key] = (Optional[List[Any]], Field(default_factory=list))
                    elif isinstance(sub_value, dict):
                        nested_fields[sub_key] = (Optional[Dict[str, Any]], Field(default_factory=dict))
                    else:
                        nested_fields[sub_key] = (Optional[str], None)
                
                # Create nested model with all fields optional
                model_classes[f"{key.capitalize()}Model"] = create_model(
                    f"{key.capitalize()}Model", 
                    __base__=DynamicBaseModel,
                    **nested_fields
                )
        
        # Create activity model preserving original field names from the template
        activity_fields = {}
        required_fields = []  # Track which fields are required
        
        for key, value in activity_schema.items():
            # Keep original field names from the template
            # Handle nested models
            if isinstance(value, dict):
                model_name = f"{key.capitalize()}Model"
                if model_name in model_classes:
                    activity_fields[key] = (
                        Optional[model_classes[model_name]], 
                        Field(default_factory=model_classes[model_name])
                    )
            # Handle lists of objects
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # For lists of objects, create a sub-model
                item_fields = {}
                for item_key, item_value in value[0].items():
                    item_fields[item_key] = (Optional[str], None)
                
                item_model = create_model(
                    f"{key.capitalize()}ItemModel",
                    __base__=DynamicBaseModel,
                    **item_fields
                )
                model_classes[f"{key.capitalize()}ItemModel"] = item_model
                activity_fields[key] = (Optional[List[item_model]], Field(default_factory=list))
            # Handle special case for dishes which might have different structures
            elif key == "dishes":
                # Create a flexible dishes model that handles different formats
                dishes_model = create_model(
                    "DishesModel",
                    __base__=DynamicBaseModel,
                    explicitly_mentioned=Field(default_factory=list),
                    visually_shown=Field(default_factory=list)
                )
                model_classes["DishesModel"] = dishes_model
                activity_fields[key] = (
                    Union[dishes_model, List[Any], Dict[str, Any]], 
                    Field(default_factory=dishes_model)
                )
            # Handle regular fields with aliases
            else:
                # Check if any field should be required (for backward compatibility)
                # Only place_name and genre are required for backward compatibility
                if key in ['place_name', 'genre']:
                    required_fields.append(key)
                    activity_fields[key] = (str, ...)
                else:
                    # All other fields are optional
                    activity_fields[key] = (Optional[str], None)
        
        # Add a catch-all field for template-specific data
        activity_fields['custom_data'] = (Optional[Dict[str, Any]], Field(default_factory=dict))
        
        # Create the activity model
        activity_model = create_model(
            "ActivityModel", 
            __base__=DynamicBaseModel,
            **activity_fields
        )
        model_classes["ActivityModel"] = activity_model
    
    # Create ClassificationResults model
    classification_fields = {}
    if 'classification_tree' in template_data:
        for level_key in template_data['classification_tree'].keys():
            field_name = level_key.replace('_context', '').replace('_type', '').replace('_details', '')
            classification_fields[field_name] = (Optional[str], None)
    
    # Also add fields for storing full classification path
    classification_fields['primary_category'] = (Optional[str], None)
    classification_fields['full_path'] = (Optional[List[str]], Field(default_factory=list))
    
    classification_model = create_model(
        "ClassificationResults",
        __base__=DynamicBaseModel,
        **classification_fields
    )
    model_classes["ClassificationResults"] = classification_model
    
    # Create the compilation model
    compilation_fields = {
        'content_type': (str, template_id),
        'classification_results': (
            Optional[classification_model], 
            Field(default_factory=classification_model)
        )
    }
    
    # Add activities field if we have an activity model
    if "ActivityModel" in model_classes:
        compilation_fields['activities'] = (
            List[model_classes["ActivityModel"]], 
            Field(default_factory=list)
        )
    else:
        # Fallback for templates without activities
        compilation_fields['activities'] = (List[Dict[str, Any]], Field(default_factory=list))
    
    # Create the compilation model
    compilation_model = create_model(
        "CompilationModel", 
        __base__=DynamicBaseModel,
        **compilation_fields
    )
    model_classes["CompilationModel"] = compilation_model
    
    return model_classes


def normalize_llm_response(data: Dict[str, Any], template_data: Dict[str, Any], field_mappings: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Normalize field names in LLM response to match expected model field names from the template.
    This is a major improvement - now we adapt to the template's field names instead of hardcoding.
    
    Args:
        data: The JSON data from LLM
        template_data: The template data that defines the expected structure
        field_mappings: Additional mappings beyond the default ones
        
    Returns:
        Normalized data with fields matching the template's structure
    """
    if field_mappings is None:
        field_mappings = FIELD_MAPPINGS
    
    # Get expected fields from template output format
    expected_fields = {}
    if 'output_format' in template_data and 'activities' in template_data['output_format'] and template_data['output_format']['activities']:
        activity_schema = template_data['output_format']['activities'][0]
        for key in activity_schema.keys():
            expected_fields[key.lower()] = key  # Map lowercase to original case
    
    # Create a new dict to avoid modifying the original
    normalized = {}
    
    # Process top-level fields
    for key, value in data.items():
        # Standardize key name if it has a mapping
        std_key = key
        
        # Handle activities array
        if std_key == "activities" and isinstance(value, list):
            normalized_activities = []
            
            for activity in value:
                if isinstance(activity, dict):
                    # Normalize each activity based on template fields
                    norm_activity = {}
                    
                    for act_key, act_value in activity.items():
                        act_key_lower = act_key.lower()
                        
                        # First, try to match to an expected field from the template
                        if act_key_lower in expected_fields:
                            target_key = expected_fields[act_key_lower]
                        # If not found in expected fields, try mappings for backward compatibility
                        elif act_key in field_mappings:
                            target_key = field_mappings[act_key]
                        # If no mapping found, keep original key
                        else:
                            target_key = act_key
                        
                        # Handle nested objects recursively
                        if isinstance(act_value, dict):
                            norm_activity[target_key] = normalize_nested_dict(act_value, field_mappings)
                        # Handle nested arrays
                        elif isinstance(act_value, list) and act_value and isinstance(act_value[0], dict):
                            norm_activity[target_key] = [
                                normalize_nested_dict(item, field_mappings) 
                                for item in act_value
                            ]
                        else:
                            norm_activity[target_key] = act_value
                    
                    # Special handling for backward compatibility:
                    # If we have 'cuisine' but template expects 'genre' and no genre is provided
                    if ('genre' in expected_fields and 'cuisine' in activity and 
                            'genre' not in norm_activity and 'cuisine' not in expected_fields):
                        norm_activity['genre'] = activity['cuisine']
                    
                    # If we have 'location' but template expects 'availability'
                    if ('availability' in expected_fields and 'location' in activity and 
                            'availability' not in norm_activity):
                        norm_activity['availability'] = {'city': activity['location']}
                    
                    normalized_activities.append(norm_activity)
                else:
                    normalized_activities.append(activity)
            
            normalized[std_key] = normalized_activities
        # Handle nested dictionaries
        elif isinstance(value, dict):
            normalized[std_key] = normalize_nested_dict(value, field_mappings)
        # Handle nested lists of dictionaries
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            normalized[std_key] = [normalize_nested_dict(item, field_mappings) for item in value]
        else:
            normalized[std_key] = value
    
    return normalized


def normalize_nested_dict(data: Dict[str, Any], field_mappings: Dict[str, str]) -> Dict[str, Any]:
    """Helper function to normalize nested dictionaries."""
    normalized = {}
    for key, value in data.items():
        # Standardize key name if it has a mapping
        std_key = field_mappings.get(key, key)
        
        # Handle nested dictionaries recursively
        if isinstance(value, dict):
            normalized[std_key] = normalize_nested_dict(value, field_mappings)
        # Handle nested lists of dictionaries
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            normalized[std_key] = [normalize_nested_dict(item, field_mappings) for item in value]
        else:
            normalized[std_key] = value
    
    return normalized


class DynamicModelManager:
    """Manager for creating and using dynamic Pydantic models."""
    
    def __init__(self):
        self.model_cache = {}  # Cache models by template_id
    
    def get_models_for_template(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get or create models for a template.
        
        Args:
            template_data: Template data containing output_format
            
        Returns:
            Dictionary of model classes
        """
        template_id = template_data.get('category', 'generic')
        
        # Check cache first
        if template_id in self.model_cache:
            return self.model_cache[template_id]
        
        # Create new models
        models = create_dynamic_models_from_template(template_data)
        
        # Cache models
        self.model_cache[template_id] = models
        
        return models
    
    def validate_data(self, data: Dict[str, Any], template_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against dynamically created models with improved error handling.
        This version guarantees that required fields are present before validation.
        
        Args:
            data: Data to validate
            template_data: Template used to create models
            
        Returns:
            Validated data as dict
        """
        # Check if we have a valid template
        if not template_data:
            print("Warning: No template data provided for validation")
            return data
        
        try:
            # Get models for this template
            models = self.get_models_for_template(template_data)
            CompilationModel = models["CompilationModel"]
            
            # First normalize field names (use your existing function)
            normalized_data = normalize_llm_response(data)
            
            # CRITICAL FIX: Add required fields to each activity if missing
            if 'activities' in normalized_data:
                for activity in normalized_data['activities']:
                    # Ensure genre exists - this is required
                    if 'genre' not in activity:
                        # Look for alternatives like cuisine
                        if 'cuisine' in activity:
                            activity['genre'] = activity['cuisine']
                        elif 'type' in activity:
                            activity['genre'] = activity['type']
                        else:
                            # Try to infer from context
                            content_lowercase = str(activity).lower()
                            if any(food_term in content_lowercase for food_term in 
                                ["restaurant", "food", "meal", "breakfast", "lunch", "dinner", "cafe"]):
                                activity['genre'] = "Restaurant"
                            elif any(hotel_term in content_lowercase for hotel_term in 
                                    ["hotel", "stay", "accommodation", "resort"]):
                                activity['genre'] = "Hotel"
                            elif any(shop_term in content_lowercase for shop_term in 
                                    ["shop", "store", "mall", "boutique", "market"]):
                                activity['genre'] = "Shop"
                            else:
                                activity['genre'] = "Establishment"
                        
                        print(f"Added missing genre field to {activity.get('place_name', 'unnamed activity')}: {activity['genre']}")
                    
                    # Ensure place_name exists - also required
                    if 'place_name' not in activity:
                        if 'restaurant_name' in activity:
                            activity['place_name'] = activity['restaurant_name']
                        elif 'name' in activity:
                            activity['place_name'] = activity['name']
                        elif 'venue_name' in activity:
                            activity['place_name'] = activity['venue_name']
                        else:
                            activity['place_name'] = "Unknown Place"
                        
                        print(f"Added missing place_name field: {activity['place_name']}")
                    
                    # Ensure availability structure exists
                    if 'availability' not in activity:
                        activity['availability'] = {}
            
            # Try validation with fixed data
            try:
                validated = CompilationModel(**normalized_data)
                return validated.model_dump(exclude_none=True)
            except Exception as e:
                print(f"Validation error after adding required fields: {e}")
                print("Creating fallback object")
                
                # Create a minimal valid model with existing data
                fallback_data = {
                    "content_type": normalized_data.get("content_type", "Unknown"),
                    "activities": []
                }
                
                # Try to salvage as much data as possible
                if 'activities' in normalized_data:
                    for activity in normalized_data['activities']:
                        # Create a minimal valid activity
                        valid_activity = {
                            "place_name": activity.get("place_name", "Unknown Place"),
                            "genre": activity.get("genre", "Establishment"),
                            "availability": activity.get("availability", {})
                        }
                        
                        # Copy any other fields that might be useful
                        for key, value in activity.items():
                            if key not in valid_activity:
                                valid_activity[key] = value
                        
                        fallback_data["activities"].append(valid_activity)
                
                fallback = CompilationModel(**fallback_data)
                return fallback.model_dump(exclude_none=True)
                
        except Exception as e:
            print(f"Overall validation error: {e}")
            # Create a minimal valid model as ultimate fallback
            try:
                fallback = CompilationModel(
                    content_type="Error",
                    activities=[]
                )
                return fallback.model_dump(exclude_none=True)
            except Exception as fallback_e:
                print(f"Error creating fallback model: {fallback_e}")
                # Ultimate fallback is a dict
                return {
                    "content_type": "Error",
                    "activities": []
                }