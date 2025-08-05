"""
OpenAI API Pricing Configuration

Current pricing rates for different OpenAI models.
Rates are per 1,000 tokens as of January 2024.
"""

from typing import Dict, Optional, Tuple
from datetime import datetime


# OpenAI API Pricing (as of 2024)
# Format: {model_name: {"input": cost_per_1k_tokens, "output": cost_per_1k_tokens}}
OPENAI_PRICING = {
    # o-series models (reasoning models)
    "o4-mini": {"input": 0.0011, "output": 0.0044},  # $1.10/$4.40 per million tokens
    "o4-mini-2025-04-16": {"input": 0.0011, "output": 0.0044},
    
    # GPT-4 models
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-32k": {"input": 0.06, "output": 0.12},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4-turbo-2024-04-09": {"input": 0.01, "output": 0.03},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-2024-05-13": {"input": 0.005, "output": 0.015},
    "gpt-4o-2024-08-06": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # $0.15/$0.60 per million tokens
    "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
    
    # GPT-3.5 models  
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo-1106": {"input": 0.001, "output": 0.002},
    "gpt-3.5-turbo-instruct": {"input": 0.0015, "output": 0.002},
    
    # Embeddings
    "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
}

# Model aliases
MODEL_ALIASES = {
    "gpt-4-0613": "gpt-4",
    "gpt-4-0314": "gpt-4",
    "gpt-3.5-turbo-0613": "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301": "gpt-3.5-turbo"
}


def get_model_pricing(model_name: str) -> Optional[Dict[str, float]]:
    """
    Get pricing information for a specific model
    
    Args:
        model_name: Name of the OpenAI model
        
    Returns:
        Dictionary with 'input' and 'output' keys containing cost per 1K tokens,
        or None if model not found
    """
    # Try exact match first
    if model_name in OPENAI_PRICING:
        return OPENAI_PRICING[model_name]
    
    # Try model aliases
    aliased_name = MODEL_ALIASES.get(model_name)
    if aliased_name and aliased_name in OPENAI_PRICING:
        return OPENAI_PRICING[aliased_name]
    
    return None


def calculate_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    """
    Calculate the total cost for API usage
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_name: Name of the OpenAI model
        
    Returns:
        Total cost in USD
    """
    pricing = get_model_pricing(model_name)
    if not pricing:
        raise ValueError(f"Pricing not found for model: {model_name}")
    
    # Convert tokens to thousands for pricing calculation
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    
    return input_cost + output_cost


def get_cost_breakdown(input_tokens: int, output_tokens: int, model_name: str) -> Dict[str, float]:
    """
    Get detailed cost breakdown for API usage
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens  
        model_name: Name of the OpenAI model
        
    Returns:
        Dictionary with cost breakdown details
    """
    pricing = get_model_pricing(model_name)
    if not pricing:
        raise ValueError(f"Pricing not found for model: {model_name}")
    
    # Convert tokens to thousands for pricing calculation
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "model_name": model_name,
        "pricing_per_1k": pricing
    }


def get_supported_models() -> list:
    """Get list of supported models"""
    return list(OPENAI_PRICING.keys())


def estimate_cost_for_text(text: str, model_name: str, is_input: bool = True) -> float:
    """
    Estimate cost for a text string (rough estimation)
    
    Args:
        text: Text to estimate
        model_name: OpenAI model name
        is_input: Whether this is input (True) or output (False) text
        
    Returns:
        Estimated cost in USD
    """
    # Rough estimation: ~4 characters per token
    estimated_tokens = len(text) // 4
    
    pricing = get_model_pricing(model_name)
    if not pricing:
        raise ValueError(f"Pricing not found for model: {model_name}")
    
    rate = pricing["input"] if is_input else pricing["output"]
    return (estimated_tokens / 1000) * rate


# Pricing update history
PRICING_HISTORY = {
    "2024-01-01": {
        "note": "Initial pricing configuration",
        "updated_by": "system"
    }
}


def log_pricing_update(model_name: str, old_pricing: Dict, new_pricing: Dict, updated_by: str = "manual"):
    """Log pricing updates for audit trail"""
    timestamp = datetime.now().isoformat()
    
    if "pricing_updates" not in PRICING_HISTORY:
        PRICING_HISTORY["pricing_updates"] = []
    
    PRICING_HISTORY["pricing_updates"].append({
        "timestamp": timestamp,
        "model": model_name,
        "old_pricing": old_pricing,
        "new_pricing": new_pricing,
        "updated_by": updated_by
    }) 