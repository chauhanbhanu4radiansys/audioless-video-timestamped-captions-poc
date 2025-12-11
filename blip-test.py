import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import argparse
from pathlib import Path
import time


class BLIPImageCaptioning:
    """BLIP Image Captioning Model Implementation"""
    
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        """
        Initialize the BLIP model and processor.
        
        Args:
            model_name: Hugging Face model identifier
        """
        print(f"Loading BLIP model: {model_name}")
        start_time = time.time()
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        load_time = time.time() - start_time
        print(f"Model loaded successfully! (Time: {load_time:.2f} seconds)")
    
    def load_image_from_url(self, url):
        """
        Load an image from a URL.
        
        Args:
            url: Image URL
            
        Returns:
            PIL Image object
        """
        try:
            start_time = time.time()
            response = requests.get(url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw).convert('RGB')
            load_time = time.time() - start_time
            print(f"Image loaded from URL (Time: {load_time:.3f} seconds)")
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
            start_time = time.time()
            image = Image.open(image_path).convert('RGB')
            load_time = time.time() - start_time
            print(f"Image loaded from path (Time: {load_time:.3f} seconds)")
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
        start_time = time.time()
        inputs = self.processor(image, prompt_text, return_tensors="pt")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        elapsed = time.time() - start_time
        print(f"Conditional caption generation time: {elapsed:.3f} seconds")
        return caption
    
    def generate_caption_unconditional(self, image):
        """
        Generate a caption for an image without any prompt.
        
        Args:
            image: PIL Image object
            
        Returns:
            Generated caption string
        """
        start_time = time.time()
        inputs = self.processor(image, return_tensors="pt")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        elapsed = time.time() - start_time
        print(f"Unconditional caption generation time: {elapsed:.3f} seconds")
        return caption


def main():
    """Main function to demonstrate BLIP image captioning"""
    
    total_start_time = time.time()
    
    # Initialize the model
    model_start_time = time.time()
    blip_model = BLIPImageCaptioning()
    model_load_time = time.time() - model_start_time
    
    # Example: Load image from URL
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    print(f"\nLoading image from URL: {img_url}")
    image_load_start = time.time()
    raw_image = blip_model.load_image_from_url(img_url)
    image_load_time = time.time() - image_load_start
    
    # Conditional image captioning
    print("\n--- Conditional Image Captioning ---")
    text = "a photography of"
    conditional_start = time.time()
    conditional_caption = blip_model.generate_caption_conditional(raw_image, text)
    conditional_time = time.time() - conditional_start
    print(f"Prompt: '{text}'")
    print(f"Caption: {conditional_caption}")
    
    # Unconditional image captioning
    print("\n--- Unconditional Image Captioning ---")
    unconditional_start = time.time()
    unconditional_caption = blip_model.generate_caption_unconditional(raw_image)
    unconditional_time = time.time() - unconditional_start
    print(f"Caption: {unconditional_caption}")
    
    total_time = time.time() - total_start_time
    
    # Print timing summary
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"Model loading time:     {model_load_time:.3f} seconds")
    print(f"Image loading time:     {image_load_time:.3f} seconds")
    print(f"Conditional caption:    {conditional_time:.3f} seconds")
    print(f"Unconditional caption:  {unconditional_time:.3f} seconds")
    print("-" * 60)
    print(f"Total time:             {total_time:.3f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLIP Image Captioning")
    parser.add_argument(
        "--image-url",
        type=str,
        default='https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg',
        help="URL of the image to caption"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Local path to the image file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photography of",
        help="Prompt text for conditional captioning"
    )
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="Use conditional captioning (requires --prompt)"
    )
    parser.add_argument(
        "--unconditional",
        action="store_true",
        help="Use unconditional captioning"
    )
    
    args = parser.parse_args()
    
    total_start_time = time.time()
    
    # Initialize model
    model_start_time = time.time()
    blip_model = BLIPImageCaptioning()
    model_load_time = time.time() - model_start_time
    
    # Load image
    image_load_start = time.time()
    if args.image_path:
        if not Path(args.image_path).exists():
            print(f"Error: Image file not found: {args.image_path}")
            exit(1)
        image = blip_model.load_image_from_path(args.image_path)
        print(f"Loaded image from: {args.image_path}")
    else:
        image = blip_model.load_image_from_url(args.image_url)
        print(f"Loaded image from URL: {args.image_url}")
    image_load_time = time.time() - image_load_start
    
    # Generate captions
    conditional_time = 0
    unconditional_time = 0
    
    if args.conditional:
        conditional_start = time.time()
        caption = blip_model.generate_caption_conditional(image, args.prompt)
        conditional_time = time.time() - conditional_start
        print(f"\nConditional Caption (prompt: '{args.prompt}'): {caption}")
    
    if args.unconditional:
        unconditional_start = time.time()
        caption = blip_model.generate_caption_unconditional(image)
        unconditional_time = time.time() - unconditional_start
        print(f"\nUnconditional Caption: {caption}")
    
    # If no specific mode is selected, run both
    if not args.conditional and not args.unconditional:
        print("\n--- Conditional Image Captioning ---")
        conditional_start = time.time()
        conditional_caption = blip_model.generate_caption_conditional(image, args.prompt)
        conditional_time = time.time() - conditional_start
        print(f"Prompt: '{args.prompt}'")
        print(f"Caption: {conditional_caption}")
        
        print("\n--- Unconditional Image Captioning ---")
        unconditional_start = time.time()
        unconditional_caption = blip_model.generate_caption_unconditional(image)
        unconditional_time = time.time() - unconditional_start
        print(f"Caption: {unconditional_caption}")
    
    total_time = time.time() - total_start_time
    
    # Print timing summary
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"Model loading time:     {model_load_time:.3f} seconds")
    print(f"Image loading time:     {image_load_time:.3f} seconds")
    if conditional_time > 0:
        print(f"Conditional caption:    {conditional_time:.3f} seconds")
    if unconditional_time > 0:
        print(f"Unconditional caption:  {unconditional_time:.3f} seconds")
    print("-" * 60)
    print(f"Total time:             {total_time:.3f} seconds")
    print("=" * 60)
