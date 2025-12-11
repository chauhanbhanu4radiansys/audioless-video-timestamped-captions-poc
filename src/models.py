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
    Optimized for CPU inference speed.
    
    Returns:
        Tuple of (processor, model) or (None, None) if not available
    """
    if BlipProcessor is None or BlipForConditionalGeneration is None:
        return None, None
    
    try:
        print("Loading BLIP captioning model...")
        
        # Optimize CPU performance
        if not torch.cuda.is_available():
            # Set optimal number of threads for CPU
            num_threads = os.cpu_count() or 4
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
            print(f"CPU optimization: Using {num_threads} threads")
        
        # Load processor with fast inference enabled
        try:
            processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-large",
                use_fast=True
            )
            print("Using fast image processor for faster inference")
        except Exception as fast_error:
            # Fallback to slow processor if fast is not available
            print(f"Fast processor not available, using slow processor: {fast_error}")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        
        # Set model to evaluation mode for faster inference
        model.eval()
        
        # Move to device
        if torch.cuda.is_available():
            model.to("cuda:0")
            print("BLIP model loaded on CUDA")
        else:
            model.to("cpu")
            print("BLIP model loaded on CPU")
        
        # Try to compile model for faster inference (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile') and not torch.cuda.is_available():
                print("Compiling model for faster CPU inference...")
                model = torch.compile(model, mode='reduce-overhead')
                print("Model compiled successfully")
        except Exception as compile_error:
            print(f"Model compilation skipped: {compile_error}")
        
        print("BLIP model loaded successfully")
        return processor, model
    except Exception as e:
        print(f"Could not load captioning model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

