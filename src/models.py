"""
Model loading for BLIP
"""
import os
import torch
from typing import Tuple, Optional

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:
    BlipProcessor = None
    BlipForConditionalGeneration = None
    print("Warning: Transformers not available. Install with: pip install transformers")


def load_captioning_model() -> Tuple[Optional[any], Optional[any]]:
    """
    Loads the BLIP image captioning model and processor.
    
    Returns:
        Tuple of (processor, model) or (None, None) if not available
    """
    if BlipProcessor is None or BlipForConditionalGeneration is None:
        return None, None
    
    try:
        print("Loading BLIP captioning model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        
        if torch.cuda.is_available():
            model.to("cuda:0")
        
        print("BLIP model loaded successfully")
        return processor, model
    except Exception as e:
        print(f"Could not load captioning model: {e}")
        return None, None
