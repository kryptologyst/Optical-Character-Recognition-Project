"""
Modern OCR Engine with Multiple Backends

This module provides a unified interface for optical character recognition
using multiple state-of-the-art OCR engines including Tesseract, EasyOCR,
and PaddleOCR.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image
import pytesseract
from loguru import logger

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available. Install with: pip install easyocr")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logger.warning("PaddleOCR not available. Install with: pip install paddleocr")


class OCREngine(Enum):
    """Available OCR engines."""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"


@dataclass
class OCRResult:
    """Container for OCR results."""
    text: str
    confidence: float
    bounding_boxes: List[Tuple[int, int, int, int]]
    engine: OCREngine
    processing_time: float


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""
    
    @abstractmethod
    def extract_text(self, image: Union[str, Path, np.ndarray]) -> OCRResult:
        """Extract text from image."""
        pass
    
    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        pass


class TesseractEngine(BaseOCREngine):
    """Tesseract OCR engine implementation."""
    
    def __init__(self, language: str = 'eng', config: str = '--psm 6'):
        """
        Initialize Tesseract engine.
        
        Args:
            language: Language code for OCR
            config: Tesseract configuration string
        """
        self.language = language
        self.config = config
        logger.info(f"Initialized Tesseract engine with language: {language}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for Tesseract OCR.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.medianBlur(thresh, 3)
        
        return denoised
    
    def extract_text(self, image: Union[str, Path, np.ndarray]) -> OCRResult:
        """
        Extract text using Tesseract.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            OCRResult containing extracted text and metadata
        """
        import time
        start_time = time.time()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image_path = str(image)
            img_array = cv2.imread(image_path)
            if img_array is None:
                raise ValueError(f"Could not load image from {image_path}")
        else:
            img_array = image.copy()
        
        # Preprocess image
        processed_img = self.preprocess_image(img_array)
        
        # Extract text
        text = pytesseract.image_to_string(
            processed_img, 
            lang=self.language, 
            config=self.config
        ).strip()
        
        # Get bounding boxes
        boxes = pytesseract.image_to_boxes(
            processed_img, 
            lang=self.language, 
            config=self.config
        )
        
        bounding_boxes = []
        for line in boxes.splitlines():
            parts = line.split()
            if len(parts) >= 6:
                x1, y1, x2, y2 = map(int, parts[1:5])
                bounding_boxes.append((x1, y1, x2, y2))
        
        processing_time = time.time() - start_time
        
        return OCRResult(
            text=text,
            confidence=0.8,  # Tesseract doesn't provide easy confidence scores
            bounding_boxes=bounding_boxes,
            engine=OCREngine.TESSERACT,
            processing_time=processing_time
        )


class EasyOCREngine(BaseOCREngine):
    """EasyOCR engine implementation."""
    
    def __init__(self, languages: List[str] = ['en'], gpu: bool = False):
        """
        Initialize EasyOCR engine.
        
        Args:
            languages: List of language codes
            gpu: Whether to use GPU acceleration
        """
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR not available. Install with: pip install easyocr")
        
        self.languages = languages
        self.gpu = gpu
        self.reader = easyocr.Reader(languages, gpu=gpu)
        logger.info(f"Initialized EasyOCR engine with languages: {languages}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        EasyOCR handles preprocessing internally.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Same image (no preprocessing needed)
        """
        return image
    
    def extract_text(self, image: Union[str, Path, np.ndarray]) -> OCRResult:
        """
        Extract text using EasyOCR.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            OCRResult containing extracted text and metadata
        """
        import time
        start_time = time.time()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image_path = str(image)
            img_array = cv2.imread(image_path)
            if img_array is None:
                raise ValueError(f"Could not load image from {image_path}")
        else:
            img_array = image.copy()
        
        # Extract text and bounding boxes
        results = self.reader.readtext(img_array)
        
        text_parts = []
        bounding_boxes = []
        confidences = []
        
        for (bbox, text, confidence) in results:
            text_parts.append(text)
            confidences.append(confidence)
            
            # Convert bbox to (x1, y1, x2, y2) format
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
            bounding_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        full_text = ' '.join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        processing_time = time.time() - start_time
        
        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            bounding_boxes=bounding_boxes,
            engine=OCREngine.EASYOCR,
            processing_time=processing_time
        )


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR engine implementation."""
    
    def __init__(self, lang: str = 'en', use_angle_cls: bool = True, use_gpu: bool = False):
        """
        Initialize PaddleOCR engine.
        
        Args:
            lang: Language code
            use_angle_cls: Whether to use angle classification
            use_gpu: Whether to use GPU acceleration
        """
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR not available. Install with: pip install paddleocr")
        
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.ocr = PaddleOCR(
            use_angle_cls=use_angle_cls, 
            lang=lang, 
            use_gpu=use_gpu
        )
        logger.info(f"Initialized PaddleOCR engine with language: {lang}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        PaddleOCR handles preprocessing internally.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Same image (no preprocessing needed)
        """
        return image
    
    def extract_text(self, image: Union[str, Path, np.ndarray]) -> OCRResult:
        """
        Extract text using PaddleOCR.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            OCRResult containing extracted text and metadata
        """
        import time
        start_time = time.time()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image_path = str(image)
            img_array = cv2.imread(image_path)
            if img_array is None:
                raise ValueError(f"Could not load image from {image_path}")
        else:
            img_array = image.copy()
        
        # Extract text
        results = self.ocr.ocr(img_array, cls=self.use_angle_cls)
        
        text_parts = []
        bounding_boxes = []
        confidences = []
        
        if results and results[0]:
            for line in results[0]:
                if line:
                    bbox, (text, confidence) = line
                    text_parts.append(text)
                    confidences.append(confidence)
                    
                    # Convert bbox to (x1, y1, x2, y2) format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)
                    bounding_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        full_text = ' '.join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        processing_time = time.time() - start_time
        
        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            bounding_boxes=bounding_boxes,
            engine=OCREngine.PADDLEOCR,
            processing_time=processing_time
        )


class OCREngineManager:
    """Manager class for multiple OCR engines."""
    
    def __init__(self, default_engine: OCREngine = OCREngine.TESSERACT):
        """
        Initialize OCR engine manager.
        
        Args:
            default_engine: Default engine to use
        """
        self.engines: Dict[OCREngine, BaseOCREngine] = {}
        self.default_engine = default_engine
        self._initialize_engines()
    
    def _initialize_engines(self) -> None:
        """Initialize available OCR engines."""
        # Always initialize Tesseract
        try:
            self.engines[OCREngine.TESSERACT] = TesseractEngine()
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {e}")
        
        # Initialize EasyOCR if available
        if EASYOCR_AVAILABLE:
            try:
                self.engines[OCREngine.EASYOCR] = EasyOCREngine()
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
        
        # Initialize PaddleOCR if available
        if PADDLEOCR_AVAILABLE:
            try:
                self.engines[OCREngine.PADDLEOCR] = PaddleOCREngine()
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
    
    def extract_text(
        self, 
        image: Union[str, Path, np.ndarray], 
        engine: Optional[OCREngine] = None
    ) -> OCRResult:
        """
        Extract text using specified or default engine.
        
        Args:
            image: Image path or numpy array
            engine: OCR engine to use (uses default if None)
            
        Returns:
            OCRResult containing extracted text and metadata
        """
        if engine is None:
            engine = self.default_engine
        
        if engine not in self.engines:
            raise ValueError(f"Engine {engine} not available")
        
        return self.engines[engine].extract_text(image)
    
    def extract_text_multiple_engines(
        self, 
        image: Union[str, Path, np.ndarray]
    ) -> Dict[OCREngine, OCRResult]:
        """
        Extract text using all available engines.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            Dictionary mapping engine names to OCR results
        """
        results = {}
        for engine_name, engine in self.engines.items():
            try:
                results[engine_name] = engine.extract_text(image)
            except Exception as e:
                logger.error(f"Error with {engine_name}: {e}")
        
        return results
    
    def get_available_engines(self) -> List[OCREngine]:
        """Get list of available engines."""
        return list(self.engines.keys())
    
    def compare_results(
        self, 
        image: Union[str, Path, np.ndarray]
    ) -> Dict[str, any]:
        """
        Compare results from all available engines.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            Comparison results
        """
        results = self.extract_text_multiple_engines(image)
        
        comparison = {
            'engines_used': list(results.keys()),
            'results': results,
            'text_lengths': {engine.value: len(result.text) for engine, result in results.items()},
            'confidences': {engine.value: result.confidence for engine, result in results.items()},
            'processing_times': {engine.value: result.processing_time for engine, result in results.items()}
        }
        
        return comparison
