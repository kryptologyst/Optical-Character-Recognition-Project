#!/usr/bin/env python3
"""
Command-line interface for OCR processing.
"""

import argparse
import sys
from pathlib import Path
import cv2
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ocr_engine import OCREngineManager, OCREngine
from config import config_manager, setup_logging
from data_generator import SyntheticTextGenerator


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Modern OCR Engine - Extract text from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py image.png                    # Extract text from image
  python cli.py image.png --engine easyocr   # Use specific engine
  python cli.py image.png --compare          # Compare all engines
  python cli.py --generate-samples           # Generate sample images
  python cli.py --config                     # Show current configuration
        """
    )
    
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Path to image file for OCR processing"
    )
    
    parser.add_argument(
        "--engine",
        choices=["tesseract", "easyocr", "paddleocr"],
        help="OCR engine to use"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare results from all available engines"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: print to stdout)"
    )
    
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate sample images for testing"
    )
    
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show current configuration"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    config = config_manager.get_config()
    if args.verbose:
        config.log_level = "DEBUG"
    setup_logging(config)
    
    # Handle configuration display
    if args.config:
        print("Current Configuration:")
        print(f"Default Engine: {config.default_engine}")
        print(f"Tesseract Language: {config.tesseract_language}")
        print(f"EasyOCR Languages: {config.easyocr_languages}")
        print(f"PaddleOCR Language: {config.paddleocr_language}")
        print(f"Image Preprocessing: {config.image_preprocessing}")
        print(f"Log Level: {config.log_level}")
        return
    
    # Handle sample generation
    if args.generate_samples:
        print("Generating sample images...")
        generator = SyntheticTextGenerator()
        dataset = generator.create_sample_images()
        print(f"Generated {len(dataset)} sample images in data/synthetic/")
        return
    
    # Validate image path
    if not args.image_path:
        parser.error("Image path is required for OCR processing")
    
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file '{image_path}' not found")
        sys.exit(1)
    
    # Initialize OCR manager
    try:
        manager = OCREngineManager()
    except Exception as e:
        print(f"Error initializing OCR manager: {e}")
        sys.exit(1)
    
    # Load image
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image '{image_path}'")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    # Process image
    try:
        if args.compare:
            print("Comparing all available engines...")
            results = manager.extract_text_multiple_engines(image)
            
            if args.format == "json":
                output_data = {}
                for engine, result in results.items():
                    output_data[engine.value] = {
                        "text": result.text,
                        "confidence": result.confidence,
                        "processing_time": result.processing_time,
                        "bounding_boxes": result.bounding_boxes
                    }
                
                output = json.dumps(output_data, indent=2)
            else:
                output_lines = []
                for engine, result in results.items():
                    output_lines.append(f"=== {engine.value.upper()} ===")
                    output_lines.append(f"Text: {result.text}")
                    output_lines.append(f"Confidence: {result.confidence:.3f}")
                    output_lines.append(f"Processing Time: {result.processing_time:.3f}s")
                    output_lines.append(f"Text Length: {len(result.text)} characters")
                    output_lines.append("")
                
                output = "\n".join(output_lines)
        else:
            # Single engine processing
            engine = None
            if args.engine:
                engine = OCREngine(args.engine)
            
            print(f"Processing with {engine.value if engine else 'default'} engine...")
            result = manager.extract_text(image, engine)
            
            if args.format == "json":
                output_data = {
                    "text": result.text,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                    "engine": result.engine.value,
                    "bounding_boxes": result.bounding_boxes
                }
                output = json.dumps(output_data, indent=2)
            else:
                output_lines = [
                    f"Engine: {result.engine.value}",
                    f"Confidence: {result.confidence:.3f}",
                    f"Processing Time: {result.processing_time:.3f}s",
                    f"Text Length: {len(result.text)} characters",
                    "",
                    "Extracted Text:",
                    result.text
                ]
                output = "\n".join(output_lines)
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(output)
            print(f"Results saved to {output_path}")
        else:
            print(output)
    
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
