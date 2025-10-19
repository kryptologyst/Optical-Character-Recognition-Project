"""
Synthetic data generation for OCR testing and demonstration.
"""

import random
import string
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
from loguru import logger


class SyntheticTextGenerator:
    """Generates synthetic text images for OCR testing."""
    
    def __init__(self, output_dir: str = "data/synthetic"):
        """
        Initialize synthetic text generator.
        
        Args:
            output_dir: Directory to save generated images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample texts for different scenarios
        self.sample_texts = {
            'simple': [
                "Hello World",
                "Python Programming",
                "Machine Learning",
                "Computer Vision",
                "Artificial Intelligence"
            ],
            'numbers': [
                "1234567890",
                "Phone: 555-123-4567",
                "Price: $29.99",
                "Date: 2024-01-15",
                "ID: ABC123XYZ"
            ],
            'mixed': [
                "Invoice #INV-2024-001",
                "Customer: John Doe",
                "Amount: $1,234.56",
                "Due Date: March 15, 2024",
                "Status: PAID"
            ],
            'paragraphs': [
                "This is a sample paragraph for testing OCR accuracy. It contains multiple sentences with various punctuation marks and numbers like 123.",
                "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet at least once.",
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
            ]
        }
    
    def generate_random_text(self, length: int = 50) -> str:
        """
        Generate random text of specified length.
        
        Args:
            length: Length of text to generate
            
        Returns:
            Random text string
        """
        return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))
    
    def create_text_image(
        self,
        text: str,
        font_size: int = 24,
        image_size: Tuple[int, int] = (400, 100),
        background_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        font_path: Optional[str] = None
    ) -> Image.Image:
        """
        Create an image with text.
        
        Args:
            text: Text to render
            font_size: Font size
            image_size: Image dimensions (width, height)
            background_color: Background color (R, G, B)
            text_color: Text color (R, G, B)
            font_path: Path to font file
            
        Returns:
            PIL Image with text
        """
        # Create image
        img = Image.new('RGB', image_size, background_color)
        draw = ImageDraw.Draw(img)
        
        # Try to load font, fallback to default
        try:
            if font_path and Path(font_path).exists():
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        
        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (image_size[0] - text_width) // 2
        y = (image_size[1] - text_height) // 2
        
        # Draw text
        draw.text((x, y), text, fill=text_color, font=font)
        
        return img
    
    def add_noise(self, image: Image.Image, noise_level: float = 0.1) -> Image.Image:
        """
        Add noise to image.
        
        Args:
            image: Input PIL image
            noise_level: Amount of noise (0-1)
            
        Returns:
            Noisy image
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        noisy_img = img_array + noise
        
        # Clip values to valid range
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_img)
    
    def add_rotation(self, image: Image.Image, angle: float = 0) -> Image.Image:
        """
        Rotate image by specified angle.
        
        Args:
            image: Input PIL image
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image
        """
        if angle == 0:
            return image
        
        return image.rotate(angle, expand=True, fillcolor=(255, 255, 255))
    
    def add_blur(self, image: Image.Image, blur_radius: int = 1) -> Image.Image:
        """
        Add blur to image.
        
        Args:
            image: Input PIL image
            blur_radius: Blur radius
            
        Returns:
            Blurred image
        """
        if blur_radius <= 0:
            return image
        
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    def generate_dataset(
        self,
        num_images: int = 50,
        categories: List[str] = None,
        variations: List[str] = None
    ) -> List[Tuple[str, str]]:
        """
        Generate a dataset of synthetic text images.
        
        Args:
            num_images: Number of images to generate
            categories: Text categories to use
            variations: Image variations to apply
            
        Returns:
            List of (image_path, ground_truth_text) tuples
        """
        if categories is None:
            categories = ['simple', 'numbers', 'mixed']
        
        if variations is None:
            variations = ['clean', 'noisy', 'rotated']
        
        dataset = []
        
        for i in range(num_images):
            # Select random category and text
            category = random.choice(categories)
            text = random.choice(self.sample_texts[category])
            
            # Create base image
            img = self.create_text_image(text)
            
            # Apply random variations
            variation = random.choice(variations)
            
            if variation == 'noisy':
                img = self.add_noise(img, noise_level=random.uniform(0.05, 0.2))
            elif variation == 'rotated':
                angle = random.uniform(-15, 15)
                img = self.add_rotation(img, angle)
            elif variation == 'blurred':
                img = self.add_blur(img, blur_radius=random.randint(1, 3))
            
            # Save image
            filename = f"synthetic_{category}_{i:03d}_{variation}.png"
            image_path = self.output_dir / filename
            img.save(image_path)
            
            dataset.append((str(image_path), text))
            
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{num_images} images")
        
        logger.info(f"Dataset generation complete. Saved {len(dataset)} images to {self.output_dir}")
        return dataset
    
    def create_sample_images(self) -> List[Tuple[str, str]]:
        """
        Create a small set of sample images for demonstration.
        
        Returns:
            List of (image_path, ground_truth_text) tuples
        """
        sample_dataset = []
        
        # Create clean samples
        for category, texts in self.sample_texts.items():
            for i, text in enumerate(texts[:2]):  # Take first 2 texts from each category
                img = self.create_text_image(text, font_size=28)
                filename = f"sample_{category}_{i}_clean.png"
                image_path = self.output_dir / filename
                img.save(image_path)
                sample_dataset.append((str(image_path), text))
        
        # Create noisy samples
        for i, text in enumerate(self.sample_texts['mixed'][:3]):
            img = self.create_text_image(text, font_size=24)
            img = self.add_noise(img, noise_level=0.15)
            filename = f"sample_noisy_{i}.png"
            image_path = self.output_dir / filename
            img.save(image_path)
            sample_dataset.append((str(image_path), text))
        
        # Create rotated samples
        for i, text in enumerate(self.sample_texts['numbers'][:2]):
            img = self.create_text_image(text, font_size=26)
            img = self.add_rotation(img, angle=random.uniform(-10, 10))
            filename = f"sample_rotated_{i}.png"
            image_path = self.output_dir / filename
            img.save(image_path)
            sample_dataset.append((str(image_path), text))
        
        logger.info(f"Created {len(sample_dataset)} sample images")
        return sample_dataset


def create_sample_images() -> None:
    """Create sample images for the OCR project."""
    generator = SyntheticTextGenerator()
    generator.create_sample_images()
    logger.info("Sample images created successfully")


if __name__ == "__main__":
    # Create sample images when run directly
    create_sample_images()
