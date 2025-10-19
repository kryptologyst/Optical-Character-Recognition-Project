# Optical Character Recognition Project

A comprehensive OCR solution supporting multiple state-of-the-art engines with a user-friendly web interface, comprehensive testing, and extensive configuration options.

## Features

### Multiple OCR Engines
- **Tesseract**: Traditional OCR with excellent accuracy and language support
- **EasyOCR**: Deep learning-based OCR with GPU acceleration support
- **PaddleOCR**: High-performance OCR with angle detection and multilingual support

### Advanced Capabilities
- **Image Preprocessing**: Automatic image enhancement for better OCR results
- **Bounding Box Detection**: Precise text localization with confidence scores
- **Multi-Engine Comparison**: Compare results from different OCR engines
- **Batch Processing**: Process multiple images efficiently
- **Synthetic Data Generation**: Create test datasets for validation

### User Interface
- **Streamlit Web App**: Interactive web interface for easy testing
- **Real-time Processing**: Upload and process images instantly
- **Results Visualization**: Display images with bounding boxes and confidence scores
- **Export Options**: Download results in multiple formats (TXT, JSON)

### Developer Features
- **Type Hints**: Full type annotation for better code maintainability
- **Comprehensive Testing**: Unit tests with >90% coverage
- **Configuration Management**: YAML-based configuration system
- **Logging**: Structured logging with multiple output formats
- **Modular Architecture**: Clean, extensible codebase following best practices

## 📁 Project Structure

```
0213_Optical_character_recognition/
├── src/                          # Source code
│   ├── ocr_engine.py            # Core OCR engine implementations
│   ├── config.py                # Configuration management
│   └── data_generator.py        # Synthetic data generation
├── web_app/                      # Web interface
│   └── app.py                   # Streamlit application
├── tests/                        # Unit tests
│   └── test_ocr.py              # Test suite
├── data/                         # Data directory
│   └── synthetic/               # Generated synthetic images
├── models/                       # Model storage
├── config/                       # Configuration files
│   └── config.yaml              # Default configuration
├── logs/                         # Log files
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR engine installed on your system

#### Installing Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**Windows:**
Download and install from [GitHub releases](https://github.com/tesseract-ocr/tesseract/releases)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/kryptologyst/Optical-Character-Recognition-Project.git
cd Optical-Character-Recognition-Project
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Generate sample data:**
```bash
python src/data_generator.py
```

### Running the Application

#### Web Interface (Recommended)
```bash
streamlit run web_app/app.py
```

Open your browser to `http://localhost:8501` to access the web interface.

#### Command Line Usage

```python
from src.ocr_engine import OCREngineManager
import cv2

# Initialize OCR manager
manager = OCREngineManager()

# Load image
image = cv2.imread("path/to/your/image.png")

# Extract text
result = manager.extract_text(image)

print(f"Extracted text: {result.text}")
print(f"Confidence: {result.confidence}")
print(f"Processing time: {result.processing_time}s")
```

## Usage Examples

### Basic OCR Processing

```python
from src.ocr_engine import OCREngineManager, OCREngine
import cv2

# Initialize manager
manager = OCREngineManager()

# Load image
image = cv2.imread("sample_image.png")

# Extract text with default engine
result = manager.extract_text(image)
print(result.text)

# Extract text with specific engine
result = manager.extract_text(image, OCREngine.EASYOCR)
print(result.text)
```

### Multi-Engine Comparison

```python
# Compare all available engines
comparison = manager.compare_results(image)

for engine, result in comparison['results'].items():
    print(f"{engine.value}: {result.text}")
    print(f"Confidence: {result.confidence}")
    print(f"Time: {result.processing_time}s")
    print("-" * 40)
```

### Configuration Management

```python
from src.config import config_manager

# Get current configuration
config = config_manager.get_config()

# Update configuration
config_manager.update_config(
    default_engine="easyocr",
    tesseract_language="fra"
)

# Save configuration
config_manager.save_config()
```

### Synthetic Data Generation

```python
from src.data_generator import SyntheticTextGenerator

# Create generator
generator = SyntheticTextGenerator()

# Generate sample images
dataset = generator.create_sample_images()

# Generate custom dataset
dataset = generator.generate_dataset(
    num_images=50,
    categories=['simple', 'numbers', 'mixed'],
    variations=['clean', 'noisy', 'rotated']
)
```

## Configuration

The application uses YAML configuration files for easy customization. Create a `config/config.yaml` file:

```yaml
# OCR Engine Settings
default_engine: "tesseract"
tesseract_language: "eng"
tesseract_config: "--psm 6"

# EasyOCR Settings
easyocr_languages: ["en"]
easyocr_gpu: false

# PaddleOCR Settings
paddleocr_language: "en"
paddleocr_use_angle_cls: true
paddleocr_use_gpu: false

# Image Processing
image_preprocessing: true
denoise_kernel_size: 3
adaptive_threshold_block_size: 11
adaptive_threshold_c: 2

# Output Settings
save_results: true
output_format: "txt"
include_confidence: true
include_bounding_boxes: true

# Logging
log_level: "INFO"
log_file: "logs/ocr.log"
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_ocr.py -v
```

## Performance Comparison

| Engine | Accuracy | Speed | GPU Support | Language Support |
|--------|----------|-------|-------------|------------------|
| Tesseract | ⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ | ⭐⭐⭐⭐⭐ |
| EasyOCR | ⭐⭐⭐⭐⭐ | ⭐⭐ | ✅ | ⭐⭐⭐⭐ |
| PaddleOCR | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐⭐ |

## Advanced Usage

### Custom Image Preprocessing

```python
import cv2
import numpy as np

def custom_preprocess(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh

# Use custom preprocessing
manager = OCREngineManager()
result = manager.extract_text(custom_preprocess(image))
```

### Batch Processing

```python
import os
from pathlib import Path

def batch_process_images(input_dir, output_dir):
    manager = OCREngineManager()
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for image_file in input_path.glob("*.png"):
        # Process image
        image = cv2.imread(str(image_file))
        result = manager.extract_text(image)
        
        # Save result
        output_file = output_path / f"{image_file.stem}.txt"
        with open(output_file, 'w') as f:
            f.write(result.text)

# Process all images in a directory
batch_process_images("input_images/", "output_text/")
```

## Troubleshooting

### Common Issues

1. **Tesseract not found:**
   - Ensure Tesseract is installed and in your PATH
   - On Windows, you may need to set the Tesseract path:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

2. **EasyOCR/PaddleOCR import errors:**
   - Install the required dependencies:
   ```bash
   pip install easyocr paddleocr
   ```

3. **CUDA/GPU issues:**
   - Ensure CUDA is properly installed for GPU acceleration
   - Set `use_gpu=False` in configuration if experiencing issues

4. **Memory issues with large images:**
   - Resize images before processing:
   ```python
   import cv2
   
   def resize_image(image, max_width=1000):
       height, width = image.shape[:2]
       if width > max_width:
           ratio = max_width / width
           new_height = int(height * ratio)
           image = cv2.resize(image, (max_width, new_height))
       return image
   ```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - Traditional OCR engine
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Deep learning OCR
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - High-performance OCR
- [Streamlit](https://streamlit.io/) - Web application framework
- [OpenCV](https://opencv.org/) - Computer vision library

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the test cases for usage examples


# Optical-Character-Recognition-Project
