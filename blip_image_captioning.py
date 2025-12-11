import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import argparse
from pathlib import Path
import time
import os
from typing import List, Dict


class BLIPImageCaptioning:
    """BLIP Image Captioning Model Implementation"""
    
    def __init__(self, model_name="Salesforce/blip-image-captioning-large"):
        """
        Initialize the BLIP model and processor.
        
        Args:
            model_name: Hugging Face model identifier
        """
        print(f"Loading BLIP model: {model_name}")
        try:
            # Try fast processor first
            self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
            print("Using fast image processor")
        except Exception:
            # Fallback to slow processor
            self.processor = BlipProcessor.from_pretrained(model_name)
            print("Using slow image processor")
        
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
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
    
    def generate_caption_conditional(self, image, prompt_text="a photography of"):
        """
        Generate a caption for an image with a conditional prompt.
        
        Args:
            image: PIL Image object
            prompt_text: Text prompt to condition the caption generation
            
        Returns:
            Generated caption string
        """
        inputs = self.processor(image, prompt_text, return_tensors="pt")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def generate_caption_unconditional(self, image):
        """
        Generate a caption for an image without any prompt.
        
        Args:
            image: PIL Image object
            
        Returns:
            Generated caption string
        """
        inputs = self.processor(image, return_tensors="pt")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption


def process_images_from_directory(
    images_dir: str,
    model_name: str = "Salesforce/blip-image-captioning-large",
    use_conditional: bool = False,
    prompt: str = "a photography of"
) -> List[Dict]:
    """
    Process all images from a directory and generate captions.
    
    Args:
        images_dir: Directory containing images
        model_name: BLIP model name
        use_conditional: Whether to use conditional captioning
        prompt: Prompt text for conditional captioning
    
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
    if use_conditional:
        print(f"Prompt: '{prompt}'")
    print()
    
    # Process all images
    total_start_time = time.time()
    results = []
    individual_times = []
    
    for i, image_file in enumerate(image_files, 1):
        image_start_time = time.time()
        
        try:
            # Load image
            image = blip_model.load_image_from_path(str(image_file))
            
            # Generate caption
            if use_conditional:
                caption = blip_model.generate_caption_conditional(image, prompt)
            else:
                caption = blip_model.generate_caption_unconditional(image)
            
            image_time = time.time() - image_start_time
            individual_times.append(image_time)
            
            result = {
                'image_path': str(image_file),
                'image_name': image_file.name,
                'caption': caption,
                'processing_time': image_time
            }
            results.append(result)
            
            # Print progress
            print(f"[{i}/{len(image_files)}] {image_file.name}")
            print(f"  Caption: {caption}")
            print(f"  Time: {image_time:.3f} seconds")
            print()
            
        except Exception as e:
            image_time = time.time() - image_start_time
            print(f"[{i}/{len(image_files)}] ❌ Error processing {image_file.name}: {e}")
            print(f"  Time: {image_time:.3f} seconds")
            print()
            
            result = {
                'image_path': str(image_file),
                'image_name': image_file.name,
                'caption': None,
                'error': str(e),
                'processing_time': image_time
            }
            results.append(result)
    
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
    print(f"  Total time (including model load): {model_load_time + total_time:.2f} seconds")
    print("=" * 60)
    
    return results


def main():
    """Main function - Processes all images in a directory"""
    parser = argparse.ArgumentParser(
        description="BLIP Image Captioning - Captions all images in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in 'images' directory (default)
  python blip_image_captioning.py
  
  # Process images in a specific directory
  python blip_image_captioning.py /path/to/images
  
  # Use conditional captioning
  python blip_image_captioning.py --conditional --prompt "a photography of"
  
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
            caption = blip_model.generate_caption_conditional(image, args.prompt)
            elapsed = time.time() - start_time
            print(f"\nConditional Caption (prompt: '{args.prompt}'): {caption}")
            print(f"Processing time: {elapsed:.3f} seconds")
        
        if args.unconditional:
            start_time = time.time()
            caption = blip_model.generate_caption_unconditional(image)
            elapsed = time.time() - start_time
            print(f"\nUnconditional Caption: {caption}")
            print(f"Processing time: {elapsed:.3f} seconds")
        
        if not args.conditional and not args.unconditional:
            print("\n--- Conditional Image Captioning ---")
            start_time = time.time()
            conditional_caption = blip_model.generate_caption_conditional(image, args.prompt)
            elapsed = time.time() - start_time
            print(f"Prompt: '{args.prompt}'")
            print(f"Caption: {conditional_caption}")
            print(f"Time: {elapsed:.3f} seconds")
            
            print("\n--- Unconditional Image Captioning ---")
            start_time = time.time()
            unconditional_caption = blip_model.generate_caption_unconditional(image)
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
            prompt=args.prompt
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
