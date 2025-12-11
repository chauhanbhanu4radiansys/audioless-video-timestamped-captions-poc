import requests
from PIL import Image
from transformers import BlipProcessor, Blip2ForConditionalGeneration
import argparse
from pathlib import Path
import time
import os
from typing import List, Dict
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


class BLIP2ImageCaptioning:
    """BLIP-2 Image Captioning Model Implementation (Flan-T5-XL) - Optimized for Speed"""
    
    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl"):
        """
        Initialize the BLIP-2 model and processor with optimizations.
        
        Args:
            model_name: Hugging Face model identifier (default: blip2-flan-t5-base)
        """
        print(f"Loading BLIP-2 model: {model_name}")
        
        # CPU optimization: Set thread counts before loading model
        if not torch.cuda.is_available():
            num_threads = os.cpu_count() or 4
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
            print(f"CPU optimization: Using {num_threads} threads")
        
        try:
            # Try fast processor first (requires torchvision)
            try:
                self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
                print("Using fast image processor")
            except (ImportError, Exception) as e:
                # Fallback to slow processor if torchvision not available or fast processor fails
                print(f"Fast processor not available ({str(e)[:50]}...), using slow processor")
                self.processor = BlipProcessor.from_pretrained(model_name, use_fast=False)
                print("Using slow image processor")
        except Exception as e:
            raise Exception(f"Failed to load processor: {e}. Make sure torchvision is installed: pip install torchvision")
        
        # Load BLIP-2 model
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.eval()
        
        # Move to device (GPU if available)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on: {self.device}")
        
        # Try to compile model for faster inference (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                if torch.cuda.is_available():
                    print("Compiling model for faster GPU inference...")
                    self.model = torch.compile(self.model, mode='max-autotune')
                else:
                    print("Compiling model for faster CPU inference...")
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                print("Model compiled successfully")
        except Exception as compile_error:
            print(f"Model compilation skipped: {compile_error}")
        
        print("Model loaded successfully!")
    
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
    
    def generate_caption_conditional(self, image, prompt_text="a photography of", max_length=30, num_beams=1):
        """
        Generate a caption for an image with a conditional prompt (optimized).
        
        Args:
            image: PIL Image object
            prompt_text: Text prompt to condition the caption generation
            max_length: Maximum caption length (default: 30 for speed)
            num_beams: Number of beams for beam search (1 = greedy, faster)
            
        Returns:
            Generated caption string
        """
        inputs = self.processor(image, prompt_text, return_tensors="pt").to(self.device)
        
        # Use inference_mode for faster inference
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=False  # Deterministic for speed
            )
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def generate_caption_unconditional(self, image, max_length=30, num_beams=1):
        """
        Generate a caption for an image without any prompt (optimized).
        
        Args:
            image: PIL Image object
            max_length: Maximum caption length (default: 30 for speed)
            num_beams: Number of beams for beam search (1 = greedy, faster)
            
        Returns:
            Generated caption string
        """
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # Use inference_mode for faster inference
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=False  # Deterministic for speed
            )
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def generate_captions_batch(self, images: List[Image.Image], use_conditional: bool = False, 
                                prompt: str = "a photography of", max_length=30, num_beams=1) -> List[str]:
        """
        Generate captions for multiple images in a batch (much faster).
        
        Args:
            images: List of PIL Image objects
            use_conditional: Whether to use conditional captioning
            prompt: Prompt text for conditional captioning
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            
        Returns:
            List of caption strings
        """
        if not images:
            return []
        
        # Process images in batch
        if use_conditional:
            inputs = self.processor(images=images, text=[prompt] * len(images), return_tensors="pt", padding=True).to(self.device)
        else:
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        
        # Use inference_mode for faster inference
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=False
            )
        
        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [caption.strip() for caption in captions]


def process_images_from_directory(
    images_dir: str,
    model_name: str = "Salesforce/blip2-flan-t5-xl",
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
        model_name: BLIP-2 model name
        use_conditional: Whether to use conditional captioning
        prompt: Prompt text for conditional captioning
        batch_size: Number of images to process at once (default: 8)
        max_length: Maximum caption length (default: 30 for speed)
        num_workers: Number of worker threads for parallel image loading (default: 6)
    
    Returns:
        List of dictionaries with image path, caption, and timing info
    """
    # Initialize model
    print("=" * 60)
    print("INITIALIZING BLIP-2 MODEL (Flan-T5-XL)")
    print("=" * 60)
    model_start_time = time.time()
    blip_model = BLIP2ImageCaptioning(model_name=model_name)
    model_load_time = time.time() - model_start_time
    print(f"Model loading time: {model_load_time:.2f} seconds")
    print()
    
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
    if blip_model.device == "cpu":
        if batch_size > 16:
            print(f"⚠️  Warning: Large batch size ({batch_size}) on CPU may be slow. Consider using GPU or reducing batch size.")
    elif torch.cuda.is_available():
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
    """Main function - Processes a single image or all images in a directory"""
    parser = argparse.ArgumentParser(
        description="BLIP-2 Image Captioning (Flan-T5-XL) - Captions a single image or all images in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image file
  python blip2_image_captioning.py image.jpg
  
  # Process all images in a directory
  python blip2_image_captioning.py /path/to/images
  
  # Single image with conditional captioning
  python blip2_image_captioning.py image.jpg --conditional --prompt "a photography of"
  
  # Directory with custom batch size
  python blip2_image_captioning.py images/ --batch-size 16
  
  # Save results to JSON
  python blip2_image_captioning.py images/ --output captions.json
        """
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a single image file or directory containing images to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        help="BLIP-2 model to use. Options: Salesforce/blip2-flan-t5-xl, Salesforce/blip2-opt-2.7b, Salesforce/blip2-opt-6.7b (default: blip2-flan-t5-xl)"
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
        help="Output JSON file to save results (optional, only used for directory processing)"
    )
    
    args = parser.parse_args()
    
    # Resolve input path
    input_path = Path(args.input_path).expanduser().resolve()
    
    if not input_path.exists():
        print(f"❌ Error: Path does not exist: {args.input_path}")
        exit(1)
    
    # Initialize model
    blip_model = BLIP2ImageCaptioning(model_name=args.model)
    
    # Check if input is a file or directory
    if input_path.is_file():
        # Single image file processing
        print("=" * 60)
        print("SINGLE IMAGE PROCESSING")
        print("=" * 60)
        print(f"Processing image: {input_path.name}")
        print()
        
        try:
            image = blip_model.load_image_from_path(str(input_path))
            
            if args.conditional:
                start_time = time.time()
                caption = blip_model.generate_caption_conditional(image, args.prompt, max_length=args.max_length)
                elapsed = time.time() - start_time
                print(f"Conditional Caption (prompt: '{args.prompt}'): {caption}")
                print(f"Processing time: {elapsed:.3f} seconds")
            else:
                start_time = time.time()
                caption = blip_model.generate_caption_unconditional(image, max_length=args.max_length)
                elapsed = time.time() - start_time
                print(f"Caption: {caption}")
                print(f"Processing time: {elapsed:.3f} seconds")
            
            # Save to JSON if requested
            if args.output:
                import json
                output_path = Path(args.output)
                result = {
                    'image_path': str(input_path),
                    'image_name': input_path.name,
                    'caption': caption,
                    'processing_time': elapsed
                }
                with open(output_path, 'w') as f:
                    json.dump([result], f, indent=2)
                print(f"\n✅ Result saved to: {output_path}")
        
        except Exception as e:
            print(f"\n❌ Error processing image: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    
    elif input_path.is_dir():
        # Directory processing
        try:
            results = process_images_from_directory(
                images_dir=str(input_path),
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
    
    else:
        print(f"❌ Error: Path is neither a file nor a directory: {args.input_path}")
        exit(1)


if __name__ == "__main__":
    main()
