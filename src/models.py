"""
Model loading for BLIP - Optimized for speed
"""
import os
import torch
from typing import Tuple, Optional
from PIL import Image

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:
    BlipProcessor = None
    BlipForConditionalGeneration = None
    print("Warning: Transformers not available. Install with: pip install transformers")

# Import cudnn for GPU optimizations
try:
    import torch.backends.cudnn as cudnn
except ImportError:
    cudnn = None


def load_captioning_model() -> Tuple[Optional[any], Optional[any]]:
    """
    Loads the BLIP image captioning model and processor.
    Optimized for both CPU and GPU inference speed.
    Includes FP16 for GPU, warmup, and other optimizations.
    
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
        else:
            # GPU optimizations
            if cudnn is not None:
                cudnn.benchmark = True  # Optimize for consistent input sizes
                cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
            print("GPU optimizations enabled")
        
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
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # Enable FP16 on GPU for 2x faster inference
        if device.startswith("cuda"):
            model.half()
            print("BLIP model loaded on CUDA with FP16 (2x faster)")
        else:
            print("BLIP model loaded on CPU")
        
        # Warmup model (important for transformer models)
        try:
            print("Warming up model...")
            dummy_image = Image.new("RGB", (384, 384), color="white")
            dummy_inputs = processor(images=dummy_image, return_tensors="pt").to(device)
            if device.startswith("cuda"):
                dummy_inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in dummy_inputs.items()}
            
            with torch.inference_mode():
                if device.startswith("cuda"):
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                        _ = model.generate(**dummy_inputs, max_length=5, num_beams=1)
                else:
                    _ = model.generate(**dummy_inputs, max_length=5, num_beams=1)
            print("Model warmup completed")
        except Exception as warmup_error:
            print(f"Model warmup skipped: {warmup_error}")
        
        # Try to compile model for faster inference (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                if torch.cuda.is_available():
                    print("Compiling model for faster GPU inference...")
                    model = torch.compile(model, mode='max-autotune')
                else:
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

