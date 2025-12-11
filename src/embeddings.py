"""
Description generation using BLIP
"""
from typing import List
from PIL import Image


def generate_scene_descriptions(
    pil_images: List[Image.Image], 
    processor, 
    model, 
    device: str,
    batch_size: int = 8
) -> List[str]:
    """
    Generates text descriptions for a batch of image frames using BLIP.
    Processes images in smaller batches to avoid memory issues.
    
    Args:
        pil_images: List of PIL Images
        processor: BLIP processor
        model: BLIP model
        device: Device string (for compatibility, but uses model's device)
        batch_size: Number of images to process at once (default: 8)
    
    Returns:
        List of caption strings
    """
    if not processor or not model or not pil_images:
        return [""] * len(pil_images)
    
    try:
        model_device = next(model.parameters()).device
        all_captions = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(pil_images), batch_size):
            batch_images = pil_images[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(pil_images) + batch_size - 1) // batch_size} ({len(batch_images)} images)...")
            
            try:
                inputs = processor(images=batch_images, return_tensors="pt").to(model_device)
                
                generated_ids = model.generate(**inputs, max_length=50)
                generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
                batch_captions = [caption.strip() for caption in generated_captions]
                all_captions.extend(batch_captions)
            except Exception as batch_error:
                print(f"Error processing batch {i // batch_size + 1}: {batch_error}")
                # Add empty captions for failed batch
                all_captions.extend([""] * len(batch_images))
        
        return all_captions
    except Exception as e:
        print(f"Error during batch caption generation: {e}")
        import traceback
        traceback.print_exc()
        return [""] * len(pil_images)
