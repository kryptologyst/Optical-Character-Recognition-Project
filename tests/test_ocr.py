"""
Unit tests for OCR project.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import os

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ocr_engine import (
    OCREngine, OCRResult, TesseractEngine, EasyOCREngine, 
    PaddleOCREngine, OCREngineManager, EASYOCR_AVAILABLE, PADDLEOCR_AVAILABLE
)
from config import OCRConfig, ConfigManager
from data_generator import SyntheticTextGenerator


class TestOCRResult:
    """Test OCRResult dataclass."""
    
    def test_ocr_result_creation(self):
        """Test OCRResult creation with valid data."""
        result = OCRResult(
            text="Hello World",
            confidence=0.95,
            bounding_boxes=[(10, 10, 100, 50)],
            engine=OCREngine.TESSERACT,
            processing_time=1.5
        )
        
        assert result.text == "Hello World"
        assert result.confidence == 0.95
        assert result.bounding_boxes == [(10, 10, 100, 50)]
        assert result.engine == OCREngine.TESSERACT
        assert result.processing_time == 1.5


class TestTesseractEngine:
    """Test TesseractEngine class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.engine = TesseractEngine()
    
    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.language == 'eng'
        assert self.engine.config == '--psm 6'
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        processed = self.engine.preprocess_image(test_image)
        
        assert processed.shape[:2] == test_image.shape[:2]  # Same height/width
        assert len(processed.shape) == 2  # Grayscale
    
    @patch('pytesseract.image_to_string')
    @patch('pytesseract.image_to_boxes')
    def test_extract_text_numpy_array(self, mock_boxes, mock_string):
        """Test text extraction with numpy array."""
        mock_string.return_value = "Test text"
        mock_boxes.return_value = "a 10 10 20 20 0\nb 30 30 40 40 0"
        
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = self.engine.extract_text(test_image)
        
        assert isinstance(result, OCRResult)
        assert result.text == "Test text"
        assert result.engine == OCREngine.TESSERACT
        assert len(result.bounding_boxes) == 2
    
    @patch('cv2.imread')
    @patch('pytesseract.image_to_string')
    @patch('pytesseract.image_to_boxes')
    def test_extract_text_file_path(self, mock_boxes, mock_string, mock_imread):
        """Test text extraction with file path."""
        mock_string.return_value = "Test text"
        mock_boxes.return_value = "a 10 10 20 20 0"
        mock_imread.return_value = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            result = self.engine.extract_text(tmp.name)
            
            assert isinstance(result, OCRResult)
            assert result.text == "Test text"
            assert result.engine == OCREngine.TESSERACT
        
        os.unlink(tmp.name)


class TestEasyOCREngine:
    """Test EasyOCREngine class."""
    
    def test_initialization_without_easyocr(self):
        """Test initialization when EasyOCR is not available."""
        with patch('ocr_engine.EASYOCR_AVAILABLE', False):
            with pytest.raises(ImportError):
                EasyOCREngine()
    
    @patch('ocr_engine.EASYOCR_AVAILABLE', True)
    @patch('easyocr.Reader')
    def test_initialization_with_easyocr(self, mock_reader):
        """Test initialization when EasyOCR is available."""
        mock_reader.return_value = Mock()
        
        engine = EasyOCREngine()
        
        assert engine.languages == ['en']
        assert engine.gpu is False
        mock_reader.assert_called_once_with(['en'], gpu=False)
    
    @patch('ocr_engine.EASYOCR_AVAILABLE', True)
    @patch('easyocr.Reader')
    def test_preprocess_image(self, mock_reader):
        """Test image preprocessing."""
        mock_reader.return_value = Mock()
        engine = EasyOCREngine()
        
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed = engine.preprocess_image(test_image)
        
        # EasyOCR doesn't modify the image
        np.testing.assert_array_equal(processed, test_image)
    
    @patch('ocr_engine.EASYOCR_AVAILABLE', True)
    @patch('easyocr.Reader')
    def test_extract_text(self, mock_reader):
        """Test text extraction."""
        mock_reader_instance = Mock()
        mock_reader_instance.readtext.return_value = [
            ([[10, 10], [100, 10], [100, 50], [10, 50]], "Hello", 0.95),
            ([[110, 10], [200, 10], [200, 50], [110, 50]], "World", 0.90)
        ]
        mock_reader.return_value = mock_reader_instance
        
        engine = EasyOCREngine()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = engine.extract_text(test_image)
        
        assert isinstance(result, OCRResult)
        assert result.text == "Hello World"
        assert result.confidence == 0.925  # Average of 0.95 and 0.90
        assert len(result.bounding_boxes) == 2
        assert result.engine == OCREngine.EASYOCR


class TestPaddleOCREngine:
    """Test PaddleOCREngine class."""
    
    def test_initialization_without_paddleocr(self):
        """Test initialization when PaddleOCR is not available."""
        with patch('ocr_engine.PADDLEOCR_AVAILABLE', False):
            with pytest.raises(ImportError):
                PaddleOCREngine()
    
    @patch('ocr_engine.PADDLEOCR_AVAILABLE', True)
    @patch('paddleocr.PaddleOCR')
    def test_initialization_with_paddleocr(self, mock_paddleocr):
        """Test initialization when PaddleOCR is available."""
        mock_paddleocr.return_value = Mock()
        
        engine = PaddleOCREngine()
        
        assert engine.lang == 'en'
        assert engine.use_angle_cls is True
        assert engine.use_gpu is False
        mock_paddleocr.assert_called_once_with(
            use_angle_cls=True, lang='en', use_gpu=False
        )
    
    @patch('ocr_engine.PADDLEOCR_AVAILABLE', True)
    @patch('paddleocr.PaddleOCR')
    def test_extract_text(self, mock_paddleocr):
        """Test text extraction."""
        mock_ocr_instance = Mock()
        mock_ocr_instance.ocr.return_value = [[
            [([[10, 10], [100, 10], [100, 50], [10, 50]], ("Hello", 0.95))],
            [([[110, 10], [200, 10], [200, 50], [110, 50]], ("World", 0.90))]
        ]]
        mock_paddleocr.return_value = mock_ocr_instance
        
        engine = PaddleOCREngine()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = engine.extract_text(test_image)
        
        assert isinstance(result, OCRResult)
        assert result.text == "Hello World"
        assert result.confidence == 0.925  # Average of 0.95 and 0.90
        assert len(result.bounding_boxes) == 2
        assert result.engine == OCREngine.PADDLEOCR


class TestOCREngineManager:
    """Test OCREngineManager class."""
    
    @patch('ocr_engine.TesseractEngine')
    def test_initialization(self, mock_tesseract):
        """Test manager initialization."""
        mock_tesseract.return_value = Mock()
        
        manager = OCREngineManager()
        
        assert OCREngine.TESSERACT in manager.engines
        assert manager.default_engine == OCREngine.TESSERACT
    
    @patch('ocr_engine.TesseractEngine')
    def test_get_available_engines(self, mock_tesseract):
        """Test getting available engines."""
        mock_tesseract.return_value = Mock()
        
        manager = OCREngineManager()
        engines = manager.get_available_engines()
        
        assert OCREngine.TESSERACT in engines
    
    @patch('ocr_engine.TesseractEngine')
    def test_extract_text_default_engine(self, mock_tesseract):
        """Test text extraction with default engine."""
        mock_engine = Mock()
        mock_engine.extract_text.return_value = OCRResult(
            text="Test", confidence=0.9, bounding_boxes=[],
            engine=OCREngine.TESSERACT, processing_time=1.0
        )
        mock_tesseract.return_value = mock_engine
        
        manager = OCREngineManager()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = manager.extract_text(test_image)
        
        assert isinstance(result, OCRResult)
        assert result.text == "Test"
        mock_engine.extract_text.assert_called_once_with(test_image)
    
    @patch('ocr_engine.TesseractEngine')
    def test_extract_text_specific_engine(self, mock_tesseract):
        """Test text extraction with specific engine."""
        mock_engine = Mock()
        mock_engine.extract_text.return_value = OCRResult(
            text="Test", confidence=0.9, bounding_boxes=[],
            engine=OCREngine.TESSERACT, processing_time=1.0
        )
        mock_tesseract.return_value = mock_engine
        
        manager = OCREngineManager()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = manager.extract_text(test_image, OCREngine.TESSERACT)
        
        assert isinstance(result, OCRResult)
        assert result.text == "Test"
    
    @patch('ocr_engine.TesseractEngine')
    def test_extract_text_invalid_engine(self, mock_tesseract):
        """Test text extraction with invalid engine."""
        mock_tesseract.return_value = Mock()
        
        manager = OCREngineManager()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            manager.extract_text(test_image, OCREngine.EASYOCR)


class TestOCRConfig:
    """Test OCRConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OCRConfig()
        
        assert config.default_engine == "tesseract"
        assert config.tesseract_language == "eng"
        assert config.easyocr_languages == ["en"]
        assert config.paddleocr_language == "en"
        assert config.image_preprocessing is True
        assert config.save_results is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = OCRConfig(
            default_engine="easyocr",
            tesseract_language="fra",
            easyocr_languages=["en", "fr"]
        )
        
        assert config.default_engine == "easyocr"
        assert config.tesseract_language == "fra"
        assert config.easyocr_languages == ["en", "fr"]


class TestConfigManager:
    """Test ConfigManager class."""
    
    def test_config_manager_initialization(self):
        """Test config manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            manager = ConfigManager(str(config_path))
            
            assert isinstance(manager.config, OCRConfig)
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            
            # Create manager and modify config
            manager = ConfigManager(str(config_path))
            manager.config.default_engine = "easyocr"
            
            # Save config
            manager.save_config()
            
            # Create new manager and load config
            manager2 = ConfigManager(str(config_path))
            
            assert manager2.config.default_engine == "easyocr"
    
    def test_update_config(self):
        """Test updating configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            manager = ConfigManager(str(config_path))
            
            manager.update_config(default_engine="paddleocr", tesseract_language="fra")
            
            assert manager.config.default_engine == "paddleocr"
            assert manager.config.tesseract_language == "fra"
    
    def test_reset_to_defaults(self):
        """Test resetting configuration to defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            manager = ConfigManager(str(config_path))
            
            # Modify config
            manager.config.default_engine = "easyocr"
            
            # Reset to defaults
            manager.reset_to_defaults()
            
            assert manager.config.default_engine == "tesseract"


class TestSyntheticTextGenerator:
    """Test SyntheticTextGenerator class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.temp_dir = tmpdir
            self.generator = SyntheticTextGenerator(output_dir=tmpdir)
    
    def test_initialization(self):
        """Test generator initialization."""
        assert Path(self.generator.output_dir).exists()
        assert len(self.generator.sample_texts) == 4  # simple, numbers, mixed, paragraphs
    
    def test_generate_random_text(self):
        """Test random text generation."""
        text = self.generator.generate_random_text(20)
        
        assert len(text) == 20
        assert isinstance(text, str)
    
    def test_create_text_image(self):
        """Test text image creation."""
        img = self.generator.create_text_image("Test", font_size=24, image_size=(200, 50))
        
        assert img.size == (200, 50)
        assert img.mode == 'RGB'
    
    def test_add_noise(self):
        """Test adding noise to image."""
        img = self.generator.create_text_image("Test")
        noisy_img = self.generator.add_noise(img, noise_level=0.1)
        
        assert noisy_img.size == img.size
        assert noisy_img.mode == img.mode
    
    def test_add_rotation(self):
        """Test adding rotation to image."""
        img = self.generator.create_text_image("Test")
        rotated_img = self.generator.add_rotation(img, angle=15)
        
        assert rotated_img.mode == img.mode
    
    def test_create_sample_images(self):
        """Test creating sample images."""
        dataset = self.generator.create_sample_images()
        
        assert len(dataset) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in dataset)
        
        # Check that files were created
        for image_path, text in dataset:
            assert Path(image_path).exists()
            assert isinstance(text, str)
    
    def test_generate_dataset(self):
        """Test generating a dataset."""
        dataset = self.generator.generate_dataset(
            num_images=5,
            categories=['simple', 'numbers'],
            variations=['clean', 'noisy']
        )
        
        assert len(dataset) == 5
        assert all(isinstance(item, tuple) and len(item) == 2 for item in dataset)
        
        # Check that files were created
        for image_path, text in dataset:
            assert Path(image_path).exists()
            assert isinstance(text, str)


# Integration tests
class TestIntegration:
    """Integration tests for the complete OCR pipeline."""
    
    @patch('ocr_engine.TesseractEngine')
    def test_end_to_end_ocr_pipeline(self, mock_tesseract):
        """Test complete OCR pipeline."""
        # Mock tesseract engine
        mock_engine = Mock()
        mock_engine.extract_text.return_value = OCRResult(
            text="Integration Test", confidence=0.95, bounding_boxes=[],
            engine=OCREngine.TESSERACT, processing_time=1.0
        )
        mock_tesseract.return_value = mock_engine
        
        # Create manager
        manager = OCREngineManager()
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Process image
        result = manager.extract_text(test_image)
        
        # Verify result
        assert isinstance(result, OCRResult)
        assert result.text == "Integration Test"
        assert result.confidence == 0.95
        assert result.engine == OCREngine.TESSERACT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
