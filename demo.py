#!/usr/bin/env python3
"""
Demo script for the modernized OCR project.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ocr_engine import OCREngineManager, OCREngine
from config import config_manager, setup_logging
from data_generator import SyntheticTextGenerator


def demo_basic_ocr():
    """Demonstrate basic OCR functionality."""
    print("🔍 OCR Engine Demo")
    print("=" * 50)
    
    # Setup logging
    config = config_manager.get_config()
    setup_logging(config)
    
    # Initialize OCR manager
    print("Initializing OCR manager...")
    manager = OCREngineManager()
    
    # Show available engines
    available_engines = manager.get_available_engines()
    print(f"Available engines: {[engine.value for engine in available_engines]}")
    
    # Generate sample images if they don't exist
    sample_dir = Path("data/synthetic")
    if not sample_dir.exists() or not list(sample_dir.glob("*.png")):
        print("Generating sample images...")
        generator = SyntheticTextGenerator()
        generator.create_sample_images()
    
    # Find a sample image
    sample_images = list(sample_dir.glob("*.png"))
    if not sample_images:
        print("No sample images found!")
        return
    
    sample_image = sample_images[0]
    print(f"Processing sample image: {sample_image.name}")
    
    # Process with default engine
    import cv2
    image = cv2.imread(str(sample_image))
    
    print("\n📝 Processing with default engine...")
    result = manager.extract_text(image)
    
    print(f"Engine: {result.engine.value}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Processing Time: {result.processing_time:.3f}s")
    print(f"Text Length: {len(result.text)} characters")
    print(f"\nExtracted Text:\n{result.text}")
    
    # Compare all engines if multiple available
    if len(available_engines) > 1:
        print(f"\n🔄 Comparing all engines...")
        comparison = manager.compare_results(image)
        
        print("\nComparison Results:")
        print("-" * 40)
        for engine, result in comparison['results'].items():
            print(f"{engine.value.upper()}:")
            print(f"  Text: {result.text[:50]}{'...' if len(result.text) > 50 else ''}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Time: {result.processing_time:.3f}s")
            print()


def demo_configuration():
    """Demonstrate configuration management."""
    print("\n⚙️ Configuration Demo")
    print("=" * 50)
    
    config = config_manager.get_config()
    
    print("Current Configuration:")
    print(f"Default Engine: {config.default_engine}")
    print(f"Tesseract Language: {config.tesseract_language}")
    print(f"EasyOCR Languages: {config.easyocr_languages}")
    print(f"PaddleOCR Language: {config.paddleocr_language}")
    print(f"Image Preprocessing: {config.image_preprocessing}")
    print(f"Log Level: {config.log_level}")
    
    # Update configuration
    print("\nUpdating configuration...")
    config_manager.update_config(
        tesseract_language="eng",
        image_preprocessing=True
    )
    
    print("Configuration updated successfully!")


def demo_synthetic_data():
    """Demonstrate synthetic data generation."""
    print("\n🎨 Synthetic Data Generation Demo")
    print("=" * 50)
    
    generator = SyntheticTextGenerator()
    
    print("Generating custom dataset...")
    dataset = generator.generate_dataset(
        num_images=5,
        categories=['simple', 'numbers'],
        variations=['clean', 'noisy']
    )
    
    print(f"Generated {len(dataset)} images:")
    for i, (image_path, text) in enumerate(dataset):
        print(f"  {i+1}. {Path(image_path).name} -> '{text}'")


def main():
    """Main demo function."""
    try:
        demo_basic_ocr()
        demo_configuration()
        demo_synthetic_data()
        
        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Run the web interface: streamlit run web_app/app.py")
        print("2. Use the CLI: python cli.py --help")
        print("3. Run tests: pytest tests/ -v")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
