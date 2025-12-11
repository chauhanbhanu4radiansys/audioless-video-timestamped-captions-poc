import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import argparse
from pathlib import Path
import time
import os
from typing import List, Dict
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


class BLIPImageCaptioning:
    def __init__(self, model_name="Salesforce/blip-image-captioning-large"):
        print(f"Loading BLIP model: {model_name}")

        try:
            self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        except Exception:
            self.processor = BlipProcessor.from_pretrained(model_name)

        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # 🔥 Enable FP16 on GPU
        if self.device.startswith("cuda"):
            self.model.half()
            print("Running model in FP16 mode for speed")

        # Warmup (important for transformer models)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            dummy = self.processor(
                images=Image.new("RGB", (384, 384), color="white"),
                return_tensors="pt"
            ).to(self.device)
            _ = self.model.generate(**dummy, max_length=5)

        print(f"Model loaded on {self.device} (optimized)")

    def load_image_from_url(self, url):
        """
        Load an image from a URL.
        
        Args:
            url: Image URL
            
        Returns:
            PIL Image object
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw).convert('RGB')
            return image
        except Exception as e:
            raise Exception(f"Error loading image from URL: {e}")
    
    def load_image_from_path(self, image_path):
        """
        Load an image from a local file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            raise Exception(f"Error loading image from path: {e}")
    
    def generate_caption_unconditional(self, image, max_length=30, num_beams=1):
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
                out = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=False
                )
        return self.processor.decode(out[0], skip_special_tokens=True)

    def generate_caption_conditional(self, image, prompt_text="a photography of", max_length=30, num_beams=1):
        inputs = self.processor(image, prompt_text, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
                out = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=False
                )
        return self.processor.decode(out[0], skip_special_tokens=True)

    def generate_captions_batch(
        self,
        images: List[Image.Image],
        use_conditional: bool = False,
        prompt: str = "a photography of",
        max_length: int = 30,
        num_beams: int = 1
    ) -> List[str]:

        if not images:
            return []

        # Prepare inputs ONCE
        if use_conditional:
            inputs = self.processor(
                images=images,
                text=[prompt] * len(images),
                return_tensors="pt",
                padding=True
            ).to(self.device, non_blocking=True)
        else:
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device, non_blocking=True)

        # 🔥 FP16 + inference_mode + autocast
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda"), dtype=torch.float16):
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=False,
                    early_stopping=True
                )

        # Decode on CPU
        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [c.strip() for c in captions]


def process_images_from_directory(
    images_dir: str,
    model_name: str = "Salesforce/blip-image-captioning-large",
    use_conditional: bool = False,
    prompt: str = "a photography of",
    batch_size: int = 8,
    max_length: int = 30,
    num_workers: int = None
) -> List[Dict]:
    """
    Process all images from a directory and generate captions (optimized with batching and parallel loading).
    
    Args:
        images_dir: Directory containing images
        model_name: BLIP model name
        use_conditional: Whether to use conditional captioning
        prompt: Prompt text for conditional captioning
        batch_size: Number of images to process at once (default: 8)
        max_length: Maximum caption length (default: 30 for speed)
        num_workers: Number of worker threads for parallel image loading (default: min(8, batch_size))
    
    Returns:
        List of dictionaries with image path, caption, and timing info
    """
    # Initialize model
    print("=" * 60)
    print("INITIALIZING BLIP MODEL")
    print("=" * 60)
    model_start_time = time.time()
    blip_model = BLIPImageCaptioning(model_name=model_name)
    model_load_time = time.time() - model_start_time
    print(f"Model loading time: {model_load_time:.2f} seconds")
    print()

    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / (1024**2)
        if free_mem > 10_000:   # >10GB free
            batch_size = min(batch_size, 32)
        elif free_mem > 5000:
            batch_size = min(batch_size, 16)
        else:
            batch_size = min(batch_size, 8)

    print(f"Auto-tuned batch size to: {batch_size}")

    
    # Get all image files from directory
    images_path = Path(images_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_files = [
        f for f in images_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        raise ValueError(f"No image files found in directory: {images_dir}")
    
    # Sort files for consistent processing order
    image_files.sort()
    
    print("=" * 60)
    print("PROCESSING IMAGES")
    print("=" * 60)
    print(f"Found {len(image_files)} images in '{images_dir}'")
    print(f"Caption mode: {'Conditional' if use_conditional else 'Unconditional'}")
    print(f"Batch size: {batch_size}")
    print(f"Max caption length: {max_length}")
    if use_conditional:
        print(f"Prompt: '{prompt}'")
    print()
    
    # Process images in batches
    total_start_time = time.time()
    results = []
    individual_times = []
    
    # Auto-adjust batch size based on device (but respect user's choice)
    original_batch_size = batch_size
    if blip_model.device == "cpu":
        # CPU can handle larger batches too, but be conservative
        # User can override with --batch-size if they want more
        if batch_size > 16:
            print(f"⚠️  Warning: Large batch size ({batch_size}) on CPU may be slow. Consider using GPU or reducing batch size.")
    elif torch.cuda.is_available():
        # Larger batches for GPU are more efficient
        if batch_size > 32:
            print(f"⚠️  Warning: Very large batch size ({batch_size}) may cause GPU memory issues.")
    
    print(f"Batch size: {batch_size} images per batch")
    
    # Set up parallel image loading workers
    if num_workers is None:
        num_workers = 6
    print(f"Using {num_workers} worker threads for parallel image loading")
    print()
    
    # Helper function to load a single image (for parallel execution)
    def load_image_safe(image_path: Path) -> tuple:
        """Load image and return (image_path, image or None, error or None)"""
        try:
            image = blip_model.load_image_from_path(str(image_path))
            return (image_path, image, None)
        except Exception as e:
            return (image_path, None, str(e))
    
    # Process in batches with parallel image loading
    for batch_start in range(0, len(image_files), batch_size):
        batch_end = min(batch_start + batch_size, len(image_files))
        batch_files = image_files[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(image_files) + batch_size - 1) // batch_size
        
        batch_start_time = time.time()
        
        try:
            # Load all images in batch in parallel using ThreadPoolExecutor
            load_start_time = time.time()
            batch_images_dict = {}
            batch_errors = {}
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all image loading tasks
                future_to_file = {
                    executor.submit(load_image_safe, image_file): image_file 
                    for image_file in batch_files
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_file):
                    image_file, image, error = future.result()
                    batch_images_dict[image_file] = image
                    if error:
                        batch_errors[image_file] = error
            
            load_time = time.time() - load_start_time
            
            # Reconstruct batch_images list in original order
            batch_images = []
            for image_file in batch_files:
                if image_file in batch_images_dict:
                    batch_images.append(batch_images_dict[image_file])
                    if image_file in batch_errors:
                        print(f"⚠️  Error loading {image_file.name}: {batch_errors[image_file]}")
                else:
                    batch_images.append(None)
            
            # Generate captions for batch
            valid_images = [img for img in batch_images if img is not None]
            valid_indices = [i for i, img in enumerate(batch_images) if img is not None]
            
            if valid_images:
                batch_captions = blip_model.generate_captions_batch(
                    valid_images,
                    use_conditional=use_conditional,
                    prompt=prompt,
                    max_length=max_length,
                    num_beams=1  # Greedy decoding for speed
                )
            else:
                batch_captions = []
            
            batch_time = time.time() - batch_start_time
            inference_time = batch_time - load_time
            avg_time_per_image = batch_time / len(batch_files) if batch_files else 0
            
            # Create results for batch
            caption_idx = 0
            for i, image_file in enumerate(batch_files):
                if i in valid_indices:
                    caption = batch_captions[caption_idx]
                    caption_idx += 1
                    error = batch_errors.get(image_file)
                else:
                    caption = None
                    error = batch_errors.get(image_file) or "Failed to load image"
                
                image_time = avg_time_per_image
                individual_times.append(image_time)
                
                result = {
                    'image_path': str(image_file),
                    'image_name': image_file.name,
                    'caption': caption,
                    'processing_time': image_time
                }
                if error:
                    result['error'] = error
                results.append(result)
            
            # Print batch progress (less verbose for speed)
            print(f"Batch {batch_num}/{total_batches}: {len(batch_files)} images | "
                  f"Load: {load_time:.2f}s | Inference: {inference_time:.2f}s | "
                  f"Total: {batch_time:.2f}s ({avg_time_per_image:.3f}s/img, "
                  f"{len(batch_files)/batch_time:.1f} img/s)")
            
        except Exception as e:
            print(f"❌ Error processing batch {batch_num}: {e}")
            # Add error results for this batch
            for image_file in batch_files:
                result = {
                    'image_path': str(image_file),
                    'image_name': image_file.name,
                    'caption': None,
                    'error': str(e),
                    'processing_time': 0.0
                }
                results.append(result)
            print()
    
    total_time = time.time() - total_start_time
    
    # Print summary
    print("=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {len(image_files)}")
    print(f"Successful: {sum(1 for r in results if r.get('caption') is not None)}")
    print(f"Failed: {sum(1 for r in results if r.get('caption') is None)}")
    print()
    print("Timing Statistics:")
    print(f"  Model loading time: {model_load_time:.2f} seconds")
    print(f"  Total processing time: {total_time:.2f} seconds")
    if individual_times:
        print(f"  Average time per image: {sum(individual_times) / len(individual_times):.3f} seconds")
        print(f"  Fastest image: {min(individual_times):.3f} seconds")
        print(f"  Slowest image: {max(individual_times):.3f} seconds")
        print(f"  Images per second: {len(image_files) / total_time:.2f}")
    print(f"  Total time (including model load): {model_load_time + total_time:.2f} seconds")
    print("=" * 60)
    
    return results


def main():
    """Main function - Processes all images in a directory"""
    parser = argparse.ArgumentParser(
        description="BLIP Image Captioning - Captions all images in a directory (Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in 'images' directory (default)
  python blip_image_captioning.py
  
  # Process images in a specific directory
  python blip_image_captioning.py /path/to/images
  
  # Use conditional captioning
  python blip_image_captioning.py --conditional --prompt "a photography of"
  
  # Adjust batch size for faster processing
  python blip_image_captioning.py --batch-size 16
  
  # Save results to JSON
  python blip_image_captioning.py --output captions.json
        """
    )
    parser.add_argument(
        "images_dir",
        type=str,
        nargs='?',
        default="images",
        help="Directory containing images to process (default: 'images')"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Salesforce/blip-image-captioning-large",
        choices=["Salesforce/blip-image-captioning-base", "Salesforce/blip-image-captioning-large"],
        help="BLIP model to use (default: large)"
    )
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="Use conditional captioning"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photography of",
        help="Prompt text for conditional captioning (default: 'a photography of')"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images to process in each batch (default: 8). Larger batches = faster processing but more memory. Try 16-32 for GPU, 8-16 for CPU."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=30,
        help="Maximum caption length (default: 30, shorter = faster)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker threads for parallel image loading (default: 6)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file to save results (optional)"
    )
    
    # Legacy arguments for single image processing
    parser.add_argument(
        "--image-url",
        type=str,
        default=None,
        help="URL of a single image to caption (legacy mode)"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Local path to a single image file (legacy mode)"
    )
    parser.add_argument(
        "--unconditional",
        action="store_true",
        help="Use unconditional captioning (legacy mode)"
    )
    
    args = parser.parse_args()
    
    # Legacy single image mode (only if explicitly requested)
    if args.image_url or args.image_path:
        print("=" * 60)
        print("LEGACY MODE: Single Image Processing")
        print("=" * 60)
        
        blip_model = BLIPImageCaptioning(model_name=args.model)
        
        if args.image_path:
            if not Path(args.image_path).exists():
                print(f"Error: Image file not found: {args.image_path}")
                exit(1)
            image = blip_model.load_image_from_path(args.image_path)
            print(f"Loaded image from: {args.image_path}")
        else:
            image = blip_model.load_image_from_path(args.image_url)
            print(f"Loaded image from URL: {args.image_url}")
        
        if args.conditional:
            start_time = time.time()
            caption = blip_model.generate_caption_conditional(image, args.prompt, max_length=args.max_length)
            elapsed = time.time() - start_time
            print(f"\nConditional Caption (prompt: '{args.prompt}'): {caption}")
            print(f"Processing time: {elapsed:.3f} seconds")
        
        if args.unconditional:
            start_time = time.time()
            caption = blip_model.generate_caption_unconditional(image, max_length=args.max_length)
            elapsed = time.time() - start_time
            print(f"\nUnconditional Caption: {caption}")
            print(f"Processing time: {elapsed:.3f} seconds")
        
        if not args.conditional and not args.unconditional:
            print("\n--- Conditional Image Captioning ---")
            start_time = time.time()
            conditional_caption = blip_model.generate_caption_conditional(image, args.prompt, max_length=args.max_length)
            elapsed = time.time() - start_time
            print(f"Prompt: '{args.prompt}'")
            print(f"Caption: {conditional_caption}")
            print(f"Time: {elapsed:.3f} seconds")
            
            print("\n--- Unconditional Image Captioning ---")
            start_time = time.time()
            unconditional_caption = blip_model.generate_caption_unconditional(image, max_length=args.max_length)
            elapsed = time.time() - start_time
            print(f"Caption: {unconditional_caption}")
            print(f"Time: {elapsed:.3f} seconds")
        
        return  # Exit early, don't process directory
    
    # Default mode: Process all images in directory
    try:
        results = process_images_from_directory(
            images_dir=args.images_dir,
            model_name=args.model,
            use_conditional=args.conditional,
            prompt=args.prompt,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers
        )
            
        # Save results to JSON if requested
        if args.output:
            import json
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Results saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
