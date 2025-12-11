import requests
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import argparse
from pathlib import Path
import time
import os
import sys
from typing import List, Dict
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class MiniCPMVImageCaptioning:
    """MiniCPM-V 2.6 (2.4B) Image Captioning Model Implementation - Optimized for Speed"""
    
    def __init__(self, model_name="openbmb/MiniCPM-V-2_6"):
        """
        Initialize the MiniCPM-V model and tokenizer with optimizations.
        
        Args:
            model_name: Hugging Face model identifier (default: MiniCPM-V-2_6)
        """
        print(f"Loading MiniCPM-V model: {model_name}")
        
        # Check available memory before loading (CPU only)
        if not torch.cuda.is_available():
            if HAS_PSUTIL:
                try:
                    available_memory_gb = psutil.virtual_memory().available / (1024**3)
                    total_memory_gb = psutil.virtual_memory().total / (1024**3)
                    print(f"System Memory: {available_memory_gb:.1f} GB available / {total_memory_gb:.1f} GB total")
                    
                    # MiniCPM-V 2.6 needs ~16GB+ RAM for CPU inference
                    if available_memory_gb < 12:
                        print(f"⚠️  WARNING: Low available memory ({available_memory_gb:.1f} GB)")
                        print("   MiniCPM-V 2.6 requires ~16GB+ RAM for CPU inference.")
                        print("   Consider:")
                        print("   1. Closing other applications to free memory")
                        print("   2. Using BLIP-2 instead (requires ~8GB RAM)")
                        print("   3. Using a machine with more RAM")
                        print("   4. Using GPU if available")
                        
                        response = input("\nContinue anyway? (y/N): ")
                        if response.lower() != 'y':
                            print("Exiting...")
                            sys.exit(1)
                except Exception as e:
                    print(f"Could not check memory: {e}")
            else:
                print("⚠️  Note: Install 'psutil' to check available memory before loading")
                print("   MiniCPM-V 2.6 requires ~16GB+ RAM for CPU inference")
            
            num_threads = os.cpu_count() or 4
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
            print(f"CPU optimization: Using {num_threads} threads")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("Tokenizer loaded")
        
        # Load model with optimizations
        if torch.cuda.is_available():
            # Use bfloat16 for GPU (faster and more memory efficient)
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation='sdpa',
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            self.model = self.model.eval().cuda()
        else:
            # CPU: Use memory-efficient loading with float16 if possible, fallback to float32
            # Try to use float16 first (half memory usage)
            try:
                print("Attempting to load model with float16 for reduced memory usage...")
                self.model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
                print("✓ Loaded with float16 (50% memory reduction)")
            except Exception as e:
                print(f"Float16 loading failed ({str(e)[:50]}...), trying float32 with memory optimizations...")
                # Fallback to float32 with memory optimizations
                self.model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
                print("✓ Loaded with float32")
            
            self.model = self.model.eval()
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Model loaded on: {self.device}")
        
        # Memory optimization: Clear cache after loading
        if not torch.cuda.is_available():
            import gc
            gc.collect()
            print("Memory cache cleared")
        
        # Try to compile model for faster inference (PyTorch 2.0+)
        # Skip compilation on CPU for MiniCPM-V as it can use more memory
        try:
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                print("Compiling model for faster GPU inference...")
                self.model = torch.compile(self.model, mode='max-autotune')
                print("Model compiled successfully")
            elif hasattr(torch, 'compile') and not torch.cuda.is_available():
                # Only compile on CPU if we have enough memory
                # Compilation can use extra memory, so skip it for large models on CPU
                print("Skipping CPU compilation to save memory (MiniCPM-V is memory-intensive)")
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
    
    def generate_caption_conditional(self, image, prompt_text="Describe this image in detail:", max_length=100):
        """
        Generate a caption for an image with a conditional prompt.
        
        Args:
            image: PIL Image object
            prompt_text: Text prompt/question about the image
            max_length: Maximum caption length
            
        Returns:
            Generated caption string
        """
        msgs = [{'role': 'user', 'content': [image, prompt_text]}]
        
        with torch.inference_mode():
            response = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        
        return response
    
    def generate_caption_unconditional(self, image, max_length=100):
        """
        Generate a caption for an image without any prompt.
        
        Args:
            image: PIL Image object
            max_length: Maximum caption length
            
        Returns:
            Generated caption string
        """
        # Use a generic prompt for unconditional captioning
        prompt = "Describe this image in detail:"
        return self.generate_caption_conditional(image, prompt_text=prompt, max_length=max_length)
    
    def generate_captions_batch(self, images: List[Image.Image], use_conditional: bool = False, 
                                prompt: str = "Describe this image in detail:", max_length=100) -> List[str]:
        """
        Generate captions for multiple images in a batch.
        
        Note: MiniCPM-V uses a chat interface, so we process images sequentially
        but optimize with inference_mode.
        
        Args:
            images: List of PIL Image objects
            use_conditional: Whether to use conditional captioning
            prompt: Prompt text for conditional captioning
            max_length: Maximum caption length
            
        Returns:
            List of caption strings
        """
        if not images:
            return []
        
        captions = []
        for image in images:
            if use_conditional:
                caption = self.generate_caption_conditional(image, prompt_text=prompt, max_length=max_length)
            else:
                caption = self.generate_caption_unconditional(image, max_length=max_length)
            captions.append(caption)
        
        return captions


def process_images_from_directory(
    images_dir: str,
    model_name: str = "openbmb/MiniCPM-V-2_6",
    use_conditional: bool = False,
    prompt: str = "Describe this image in detail:",
    batch_size: int = 4,
    max_length: int = 100,
    num_workers: int = None
) -> List[Dict]:
    """
    Process all images from a directory and generate captions (optimized with parallel loading).
    
    Args:
        images_dir: Directory containing images
        model_name: MiniCPM-V model name
        use_conditional: Whether to use conditional captioning
        prompt: Prompt text for conditional captioning
        batch_size: Number of images to process at once (default: 4, MiniCPM-V is memory intensive)
        max_length: Maximum caption length (default: 100)
        num_workers: Number of worker threads for parallel image loading (default: 6)
    
    Returns:
        List of dictionaries with image path, caption, and timing info
    """
    # Initialize model
    print("=" * 60)
    print("INITIALIZING MiniCPM-V MODEL")
    print("=" * 60)
    model_start_time = time.time()
    minicpm_model = MiniCPMVImageCaptioning(model_name=model_name)
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
    
    # MiniCPM-V is memory intensive, so be conservative with batch sizes
    if minicpm_model.device == "cpu":
        if batch_size > 4:
            print(f"⚠️  Warning: Large batch size ({batch_size}) on CPU may be slow. Consider using GPU or reducing batch size.")
            batch_size = min(batch_size, 4)
    elif torch.cuda.is_available():
        if batch_size > 8:
            print(f"⚠️  Warning: Very large batch size ({batch_size}) may cause GPU memory issues.")
            batch_size = min(batch_size, 8)
    
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
            image = minicpm_model.load_image_from_path(str(image_path))
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
                batch_captions = minicpm_model.generate_captions_batch(
                    valid_images,
                    use_conditional=use_conditional,
                    prompt=prompt,
                    max_length=max_length
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
            import traceback
            traceback.print_exc()
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
        description="MiniCPM-V 2.6 Image Captioning - Captions a single image or all images in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image file
  python minicpmv_image_captioning.py image.jpg
  
  # Process all images in a directory
  python minicpmv_image_captioning.py /path/to/images
  
  # Single image with conditional captioning
  python minicpmv_image_captioning.py image.jpg --conditional --prompt "What objects are visible in this image?"
  
  # Directory with custom batch size
  python minicpmv_image_captioning.py images/ --batch-size 4
  
  # Save results to JSON
  python minicpmv_image_captioning.py images/ --output captions.json
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
        default="openbmb/MiniCPM-V-2_6",
        help="MiniCPM-V model to use (default: MiniCPM-V-2_6)"
    )
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="Use conditional captioning"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail:",
        help="Prompt text/question for conditional captioning (default: 'Describe this image in detail:')"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of images to process in each batch (default: 4). MiniCPM-V is memory intensive. Try 4-8 for GPU, 2-4 for CPU."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum caption length (default: 100)"
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
    minicpm_model = MiniCPMVImageCaptioning(model_name=args.model)
    
    # Check if input is a file or directory
    if input_path.is_file():
        # Single image file processing
        print("=" * 60)
        print("SINGLE IMAGE PROCESSING")
        print("=" * 60)
        print(f"Processing image: {input_path.name}")
        print()
        
        try:
            image = minicpm_model.load_image_from_path(str(input_path))
            
            if args.conditional:
                start_time = time.time()
                caption = minicpm_model.generate_caption_conditional(image, prompt_text=args.prompt, max_length=args.max_length)
                elapsed = time.time() - start_time
                print(f"Conditional Caption (prompt: '{args.prompt}'): {caption}")
                print(f"Processing time: {elapsed:.3f} seconds")
            else:
                start_time = time.time()
                caption = minicpm_model.generate_caption_unconditional(image, max_length=args.max_length)
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
