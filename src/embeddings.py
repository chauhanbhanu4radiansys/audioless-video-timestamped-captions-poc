"""
Description generation using BLIP
"""
import torch
from typing import List
from PIL import Image


def generate_scene_descriptions(
    pil_images: List[Image.Image], 
    processor, 
    model, 
    device: str,
    batch_size: int = None
) -> List[str]:
    """
    Generates text descriptions for a batch of image frames using BLIP.
    Processes images in smaller batches to avoid memory issues.
    Optimized for CPU inference speed.
    
    Args:
        pil_images: List of PIL Images
        processor: BLIP processor
        model: BLIP model
        device: Device string (for compatibility, but uses model's device)
        batch_size: Number of images to process at once (auto-detected if None)
    
    Returns:
        List of caption strings
    """
    if not processor or not model or not pil_images:
        return [""] * len(pil_images)
    
    try:
        model_device = next(model.parameters()).device
        is_cpu = model_device.type == "cpu"
        
        # Auto-detect optimal batch size based on device
        if batch_size is None:
            if is_cpu:
                batch_size = 4  # Smaller batches for CPU
            else:
                batch_size = 8  # Larger batches for GPU
        
        all_captions = []
        total_batches = (len(pil_images) + batch_size - 1) // batch_size
        
        # Use inference_mode for faster CPU inference (PyTorch 2.0+)
        inference_context = torch.inference_mode() if hasattr(torch, 'inference_mode') else torch.no_grad()
        
        # Process in batches to avoid memory issues
        with inference_context:
            for i in range(0, len(pil_images), batch_size):
                batch_images = pil_images[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                if batch_num % 10 == 0 or batch_num == total_batches:
                    print(f"Processing batch {batch_num}/{total_batches} ({len(batch_images)} images)...")
                
                try:
                    # Process images
                    inputs = processor(images=batch_images, return_tensors="pt").to(model_device)
                    
                    # Generate with optimized settings for CPU
                    generation_kwargs = {
                        "max_length": 30,  # Reduced from 50 for faster generation
                        "num_beams": 3,    # Reduced from default for speed
                        "do_sample": False, # Deterministic for speed
                    }
                    
                    # For CPU, use greedy decoding (faster)
                    if is_cpu:
                        generation_kwargs["num_beams"] = 1
                        generation_kwargs.pop("do_sample", None)
                    
                    generated_ids = model.generate(**inputs, **generation_kwargs)
                    generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    batch_captions = [caption.strip() for caption in generated_captions]
                    all_captions.extend(batch_captions)
                    
                except Exception as batch_error:
                    print(f"Error processing batch {batch_num}: {batch_error}")
                    # Add empty captions for failed batch
                    all_captions.extend([""] * len(batch_images))
        
        return all_captions
    except Exception as e:
        print(f"Error during batch caption generation: {e}")
        import traceback
        traceback.print_exc()
        return [""] * len(pil_images)

