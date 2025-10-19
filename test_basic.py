#!/usr/bin/env python3
"""
Simple test script that works without external dependencies.
"""

import sys
from pathlib import Path

def test_project_structure():
    """Test that the project structure is correct."""
    print("🔍 Testing Project Structure")
    print("=" * 50)
    
    required_dirs = [
        "src",
        "web_app", 
        "tests",
        "data",
        "models",
        "config",
        "logs"
    ]
    
    required_files = [
        "requirements.txt",
        "README.md",
        ".gitignore",
        "cli.py",
        "demo.py",
        "src/ocr_engine.py",
        "src/config.py",
        "src/data_generator.py",
        "web_app/app.py",
        "tests/test_ocr.py",
        "config/config.yaml"
    ]
    
    print("Checking directories...")
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (missing)")
    
    print("\nChecking files...")
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} (missing)")
    
    print("\nChecking synthetic data...")
    synthetic_dir = Path("data/synthetic")
    if synthetic_dir.exists():
        sample_images = list(synthetic_dir.glob("*.png"))
        print(f"  ✅ Found {len(sample_images)} sample images")
    else:
        print("  ❌ No synthetic data directory")


def test_imports():
    """Test that modules can be imported."""
    print("\n📦 Testing Module Imports")
    print("=" * 50)
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent / "src"))
    
    try:
        from config import OCRConfig, ConfigManager
        print("  ✅ config module imported successfully")
        
        config = OCRConfig()
        print(f"  ✅ OCRConfig created with default engine: {config.default_engine}")
        
    except ImportError as e:
        print(f"  ❌ config module import failed: {e}")
    
    try:
        from data_generator import SyntheticTextGenerator
        print("  ✅ data_generator module imported successfully")
        
        generator = SyntheticTextGenerator(output_dir="temp_test")
        print("  ✅ SyntheticTextGenerator created successfully")
        
        # Clean up
        import shutil
        if Path("temp_test").exists():
            shutil.rmtree("temp_test")
            
    except ImportError as e:
        print(f"  ❌ data_generator module import failed: {e}")
    
    # Test OCR engine import (may fail if dependencies not installed)
    try:
        from ocr_engine import OCREngine, OCRResult
        print("  ✅ ocr_engine module imported successfully")
        
        result = OCRResult(
            text="Test",
            confidence=0.9,
            bounding_boxes=[],
            engine=OCREngine.TESSERACT,
            processing_time=1.0
        )
        print(f"  ✅ OCRResult created: {result.text}")
        
    except ImportError as e:
        print(f"  ⚠️  ocr_engine module import failed (expected if dependencies not installed): {e}")


def test_configuration():
    """Test configuration functionality."""
    print("\n⚙️ Testing Configuration")
    print("=" * 50)
    
    try:
        from config import OCRConfig, ConfigManager
        
        # Test default config
        config = OCRConfig()
        print(f"  ✅ Default engine: {config.default_engine}")
        print(f"  ✅ Tesseract language: {config.tesseract_language}")
        print(f"  ✅ EasyOCR languages: {config.easyocr_languages}")
        
        # Test config manager
        config_manager = ConfigManager()
        print("  ✅ ConfigManager created successfully")
        
        # Test config update
        config_manager.update_config(default_engine="test")
        print("  ✅ Configuration updated successfully")
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")


def test_synthetic_data():
    """Test synthetic data generation."""
    print("\n🎨 Testing Synthetic Data Generation")
    print("=" * 50)
    
    try:
        from data_generator import SyntheticTextGenerator
        
        generator = SyntheticTextGenerator(output_dir="temp_test_data")
        
        # Test random text generation
        random_text = generator.generate_random_text(20)
        print(f"  ✅ Generated random text: '{random_text}'")
        
        # Test text image creation
        from PIL import Image
        img = generator.create_text_image("Test Text", font_size=24)
        print(f"  ✅ Created text image: {img.size}")
        
        # Test noise addition
        noisy_img = generator.add_noise(img, noise_level=0.1)
        print(f"  ✅ Added noise to image: {noisy_img.size}")
        
        # Clean up
        import shutil
        if Path("temp_test_data").exists():
            shutil.rmtree("temp_test_data")
        
    except Exception as e:
        print(f"  ❌ Synthetic data test failed: {e}")


def main():
    """Main test function."""
    print("🚀 Modern OCR Project - Basic Tests")
    print("=" * 60)
    
    test_project_structure()
    test_imports()
    test_configuration()
    test_synthetic_data()
    
    print("\n✅ Basic tests completed!")
    print("\nTo run the full application:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Install Tesseract OCR on your system")
    print("3. Run demo: python demo.py")
    print("4. Run web app: streamlit run web_app/app.py")


if __name__ == "__main__":
    main()
