"""
Description generation using BLIP - Optimized for speed with batch processing
"""
import torch
import time
from typing import List
from PIL import Image


def generate_scene_descriptions(
    pil_images: List[Image.Image], 
    processor, 
    model, 
    device: str,
    batch_size: int = 16,
    num_workers: int = 6,
    max_length: int = 30
) -> List[str]:
    """
    Generates text descriptions for a batch of image frames using BLIP.
    Processes images in batches for optimal speed.
    Optimized for both CPU and GPU inference.
    
    Args:
        pil_images: List of PIL Images
        processor: BLIP processor
        model: BLIP model
        device: Device string (for compatibility, but uses model's device)
        batch_size: Number of images to process at once (default: 16)
        num_workers: Reserved for future use (default: 6)
        max_length: Maximum caption length (default: 30 for speed)
    
    Returns:
        List of caption strings
    """
    if not processor or not model or not pil_images:
        return [""] * len(pil_images)
    
    try:
        model_device = next(model.parameters()).device
        is_cuda = model_device.type == "cuda"
        
        # Use batch_size=16 as default (from blip_faster.py optimizations)
        if batch_size is None:
            batch_size = 16
        
        # Use num_workers=6 as default for parallel image loading
        if num_workers is None:
            num_workers = 6
        
        print(f"Processing {len(pil_images)} images with batch_size={batch_size}, num_workers={num_workers}")
        
        all_captions = []
        total_batches = (len(pil_images) + batch_size - 1) // batch_size
        
        # Helper function to resize image if needed (from blip_faster.py optimizations)
        def resize_image_if_needed(image: Image.Image, max_size: int = 384) -> Image.Image:
            """Resize image if it's too large (smaller images = faster processing)."""
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                return image.resize(new_size, Image.Resampling.LANCZOS)
            return image
        
        # Process in batches with optimized inference
        with torch.inference_mode():
            for i in range(0, len(pil_images), batch_size):
                batch_images = pil_images[i:i + batch_size]
                batch_num = i // batch_size + 1
                batch_start_time = time.time()
                
                try:
                    # Resize images for faster processing
                    resize_start = time.time()
                    resized_images = [resize_image_if_needed(img) for img in batch_images]
                    resize_time = time.time() - resize_start
                    
                    # Process images with padding for batch processing
                    prep_start = time.time()
                    inputs = processor(
                        images=resized_images,
                        return_tensors="pt",
                        padding=True
                    ).to(model_device, non_blocking=True)
                    
                    # Convert to FP16 if on GPU (for faster inference)
                    if is_cuda:
                        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                    prep_time = time.time() - prep_start
                    
                    # Generate with optimized settings (greedy decoding for speed)
                    # Note: early_stopping is not valid for BLIP model, removed to avoid warnings
                    generation_kwargs = {
                        "max_length": max_length,
                        "num_beams": 1,  # Greedy decoding for speed
                        "do_sample": False
                    }
                    
                    # Use autocast for GPU
                    inference_start = time.time()
                    if is_cuda:
                        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                            generated_ids = model.generate(**inputs, **generation_kwargs)
                    else:
                        generated_ids = model.generate(**inputs, **generation_kwargs)
                    inference_time = time.time() - inference_start
                    
                    # Decode captions
                    decode_start = time.time()
                    generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    batch_captions = [caption.strip() for caption in generated_captions]
                    decode_time = time.time() - decode_start
                    
                    all_captions.extend(batch_captions)
                    
                    # Calculate total batch time
                    batch_time = time.time() - batch_start_time
                    avg_time_per_image = batch_time / len(batch_images)
                    
                    # Print batch timing information
                    print(f"Batch {batch_num}/{total_batches}: {len(batch_images)} images | "
                          f"Resize: {resize_time:.3f}s | Prep: {prep_time:.3f}s | "
                          f"Inference: {inference_time:.3f}s | Decode: {decode_time:.3f}s | "
                          f"Total: {batch_time:.3f}s ({avg_time_per_image:.3f}s/img, "
                          f"{len(batch_images)/batch_time:.1f} img/s)")
                    
                except Exception as batch_error:
                    batch_time = time.time() - batch_start_time
                    print(f"❌ Error processing batch {batch_num}/{total_batches}: {batch_error} (took {batch_time:.3f}s)")
                    import traceback
                    traceback.print_exc()
                    # Add empty captions for failed batch
                    all_captions.extend([""] * len(batch_images))
        
        return all_captions
    except Exception as e:
        print(f"Error during batch caption generation: {e}")
        import traceback
        traceback.print_exc()
        return [""] * len(pil_images)

